"""
Simple validation test for the layer mixing fix.
Tests the core hypothesis: 70/30 gender-biased mixing at layers 8-11 reduces facial widening.
"""
import os
import sys
import json
import torch
import numpy as np
import random
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')

# Import KinshipForge modules
from kinshipforge.experiments.logger import ExperimentLogger, set_deterministic
from kinshipforge.metrics import compute_geometry_metrics, ArcFaceEvaluator

# Import StyleGene modules
import models.stylegene.api as stylegene_api
from models.stylegene.api import init_model, generate_child, brdas_sampler, tensor2rgb
from models.stylegene.gene_crossover_mutation import fuse_latent
from models.stylegene.gene_pool import GenePoolFactory
from models.stylegene.fair_face_model import init_fair_model, predict_race
from preprocess.align_images import align_face
from models.stylegene.util import load_img

import torch.nn.functional as F
import tempfile
import cv2
import pickle
import os

from configs import path_ckpt_genepool, path_ckpt_fairface, path_ckpt_landmark68

# Configuration
config = {
    "seed": 42,
    "age": "5-10",
    "gender": "male",
    "gamma": 0.05,
    "eta": 0.4,
    "arcs_lambda": 0.0,
    "father_weight": 0.5,
    "mother_weight": 0.5,
    "mix_mode": "gender_biased",
    "crossover_mode": "rfg_linear",
    "mutation_mode": "brdas"
}

test_pairs = [
    ("father_p1.jpg", "mother_p1.jpg", "male", "Indian", "Indian", "P1_Shahrukh_Gauri"),
    ("father_p2.jpg", "mother_p2.jpeg", "male", "East Asian", "East Asian", "P2_Jackie_Joan"),
    ("father_p3.jpg", "mother_p3.jpeg", "male", "Black", "Black", "P3_Obama_Michelle"),
    ("father_p4.jpg", "mother_p4.jpg", "male", "White", "White", "P4_TomHanks_Rita"),
    ("father_p5.jpg", "mother_p5.jpg", "male", "Black", "White", "P5_Ben_Laura"),
]

PHOTOS = 'C:/Users/mdiza/coding/KinshipForge-iz/archive'
POOL_AGE_MAP = {
    '5-10': '3-9',
    '11-15': '10-19',
    '16-21': '20-29'
}

variants = [
    {"name": "baseline_50_50", "geometry_weight": 0.5, "texture_weight": 0.5, "mix_mode": "fixed_50_50"},
    {"name": "male_bias_0.6", "geometry_weight": 0.6, "texture_weight": 0.5, "mix_mode": "gender_biased"},
    {"name": "male_bias_0.7", "geometry_weight": 0.7, "texture_weight": 0.5, "mix_mode": "gender_biased"},
    {"name": "male_bias_0.8", "geometry_weight": 0.8, "texture_weight": 0.5, "mix_mode": "gender_biased"},
]

seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_encode(path, encoder, mean_latent, device):
    raw = cv2.imread(path)
    if raw is None:
        raise ValueError(f"Could not load {path}")
    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    aligned = align_face(raw_rgb)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    img_t = load_img(tmp.name).to(device)
    w18 = encoder(F.interpolate(img_t, size=(256, 256))) + mean_latent
    os.unlink(tmp.name)
    return w18, aligned


def query_parent_pools(geneFactor, encoder, w2sub34, pool_age, gender, race_f, race_m):
    """Retrieve parent pools independently."""
    if race_f == race_m:
        entries = geneFactor(encoder, w2sub34, pool_age, gender, race_f)
        if not entries:
            print(f"  ⚠️ Same-race pool empty for {pool_age}-{gender}-{race_f} — expanding age bucket")
            for age in ['0-2', '3-9', '10-19', '20-29']:
                if age != pool_age:
                    entries += geneFactor(encoder, w2sub34, age, gender, race_f)
        return entries

    father_pool = geneFactor(encoder, w2sub34, pool_age, gender, race_f)
    mother_pool = geneFactor(encoder, w2sub34, pool_age, gender, race_m)

    if not father_pool:
        print(f"  ⚠️ Father pool empty for {pool_age}-{gender}-{race_f} — expanding age bucket")
        expanded = []
        for age in ['0-2', '3-9', '10-19', '20-29']:
            expanded += geneFactor(encoder, w2sub34, age, gender, race_f)
        father_pool = expanded

    if not mother_pool:
        print(f"  ⚠️ Mother pool empty for {pool_age}-{gender}-{race_m} — expanding age bucket")
        expanded = []
        for age in ['0-2', '3-9', '10-19', '20-29']:
            expanded += geneFactor(encoder, w2sub34, age, gender, race_m)
        mother_pool = expanded

    return {"father_pool": father_pool, "mother_pool": mother_pool}


def run_single_test(w18_F, w18_M, geneFactor, encoder, w2sub34, 
                    pool_age, gender, race_f, race_m, 
                    variant, seed, arcface):
    """Run a single generation with given variant."""
    set_seed(seed)
    
    # Get pools
    pools = query_parent_pools(geneFactor, encoder, w2sub34, pool_age, gender, race_f, race_m)
    
    if isinstance(pools, dict):
        random_fakes = brdas_sampler(
            pools["father_pool"], pools["mother_pool"],
            father_weight=0.5, mother_weight=0.5
        )
    else:
        random_fakes = pools
    
    # Generate child with variant
    img_C, w18_syn = generate_child(
        w18_F.clone(), w18_M.clone(), random_fakes,
        gamma=0.05,
        eta=0.4,
        arcs_lambda=0.0,
        child_gender=gender,
        geometry_weight=variant['geometry_weight'],
        texture_weight=variant['texture_weight']
    )
    
    child_np = tensor2rgb(img_C)
    
    # Metrics
    geom = compute_geometry_metrics(child_np)
    identity = arcface.identity_metrics(
        child_np,
        cv2.cvtColor(cv2.imread(f"{PHOTOS}/father_p1.jpg"), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread(f"{PHOTOS}/mother_p1.jpg"), cv2.COLOR_BGR2RGB)
    )
    
    return {
        'geometry': geom,
        'identity': identity,
        'wh_ratio': geom.get('wh_ratio', -1),
        'norm_width': geom.get('norm_width', -1),
        'norm_jaw_width': geom.get('norm_jaw_width', -1),
        'arcface_father': identity.get('arcface_father', 0),
        'arcface_mother': identity.get('arcface_mother', 0),
    }


def main():
    print("=" * 70)
    print("LAYER MIXING ABLATION - Validation Test")
    print("=" * 70)
    
    # Initialize logger
    logger = ExperimentLogger("LAYER_MIXING_VALIDATION", config)
    logger.save_config()
    
    # Load models
    print("\nLoading models...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    encoder, generator, sub2w, w2sub34, mean_latent = init_model()
    encoder = encoder.to(device)
    generator = generator.to(device)
    sub2w = sub2w.to(device)
    w2sub34 = w2sub34.to(device)
    mean_latent = mean_latent.to(device)
    
    # Inject models into api module for global access
    stylegene_api.generator = generator
    stylegene_api.w2sub34 = w2sub34
    stylegene_api.sub2w = sub2w
    
    print("Loading gene pool...")
    pool_data = torch.load(path_ckpt_genepool, map_location='cpu', weights_only=False)
    
    geneFactor = GenePoolFactory(
        root_ffhq=None,
        device=device,
        mean_latent=mean_latent,
        max_sample=300
    )
    geneFactor.pools = pool_data
    print(f"Gene pool loaded: {len(geneFactor.pools)} keys")
    
    # FairFace
    model_fair_7 = init_fair_model(device)
    print("[OK] FairFace loaded")
    
    # ArcFace
    arcface = ArcFaceEvaluator(device)
    print("[OK] ArcFace loaded")
    
    all_results = []
    
    for pair_idx, (f_file, m_file, gender, race_f, race_m, pair_name) in enumerate(test_pairs):
        print(f"\n{'='*70}")
        print(f"Pair {pair_idx+1}/{len(test_pairs)}: {pair_name}")
        print(f"{'='*70}")
        
        # Encode parents
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        w18_F, aligned_f = load_and_encode(f_path, encoder, mean_latent, device)
        w18_M, aligned_m = load_and_encode(m_path, encoder, mean_latent, device)
        
        # Race detection
        race_f_det, _, _, _ = predict_race(model_fair_7, 
            torch.from_numpy(aligned_f.transpose(2,0,1)).unsqueeze(0).float().to(device)/255.0, device)
        race_m_det, _, _, _ = predict_race(model_fair_7,
            torch.from_numpy(aligned_m.transpose(2,0,1)).unsqueeze(0).float().to(device)/255.0, device)
        print(f"  Father: {race_f_det}, Mother: {race_m_det}")
        
        for display_age, pool_age in POOL_AGE_MAP.items():
            print(f"\n  Age: {display_age} (pool: {pool_age})")
            
            for variant in variants:
                v_name = variant['name']
                g_w = variant['geometry_weight']
                t_w = variant['texture_weight']
                
                print(f"    {v_name} (geom={g_w}, tex={t_w})")
                
                seed_results = []
                for seed in seeds:
                    try:
                        res = run_single_test(w18_F, w18_M, geneFactor, encoder, w2sub34,
                                              pool_age, gender, race_f_det, race_m_det,
                                              variant, seed, arcface)
                        seed_results.append(res)
                    except Exception as e:
                        print(f"      Seed {seed} failed: {e}")
                        continue
                
                if not seed_results:
                    continue
                
                # Aggregate
                agg = {}
                for key in ['wh_ratio', 'norm_width', 'norm_jaw_width', 'arcface_father', 'arcface_mother']:
                    vals = [r[key] for r in seed_results if r[key] > 0]
                    if vals:
                        agg[f'{key}_mean'] = float(np.mean(vals))
                        agg[f'{key}_std'] = float(np.std(vals))
                        agg[f'{key}_ci95'] = float(1.96 * np.std(vals) / np.sqrt(len(vals)))
                
                result = {
                    'pair': pair_name,
                    'age': display_age,
                    'variant': v_name,
                    'geometry_weight': g_w,
                    'texture_weight': t_w,
                    'mix_mode': variant['mix_mode'],
                    **agg
                }
                all_results.append(result)
                
                # Log
                logger.log_result(v_name, pair_name, 42, display_age, agg)
                
                print(f"      wh_ratio: {agg.get('wh_ratio_mean', 'N/A'):.4f} ± {agg.get('wh_ratio_std', 'N/A'):.4f}")
                print(f"      jaw_width: {agg.get('norm_jaw_width_mean', 'N/A'):.4f} ± {agg.get('norm_jaw_width_std', 'N/A'):.4f}")
                print(f"      arcface_f: {agg.get('arcface_father_mean', 'N/A'):.4f}")
                print(f"      arcface_m: {agg.get('arcface_mother_mean', 'N/A'):.4f}")
    
    # Save results
    results_file = logger.exp_dir / 'quantitative_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate comparison report
    generate_report(all_results, logger.exp_dir)
    
    logger.log_complete(
        ExperimentRecord(
            timestamp=datetime.now().isoformat(),
            experiment_name='LAYER_MIXING_VALIDATION',
            git_commit='local',
            config_hash='test',
            random_seed=42,
            parents='{}',
            child_age='5-10',
            child_gender='male',
            gamma=0.05, eta=0.4, arcs_lambda=0.0,
            father_weight=0.5, mother_weight=0.5,
            mix_mode='gender_biased',
            crossover_mode='rfg_linear',
            mutation_mode='brdas',
            status='completed',
            metrics=json.dumps({'total_tests': len(all_results)}),
            decision='pending',
            notes='Validation complete'
        ),
        metrics={'total_tests': len(all_results)},
        decision='pending_analysis',
        notes='Layer mixing validation complete'
    )
    
    print(f"\n[OK] Experiment complete! Results in: {logger.exp_dir}")
    return all_results


def generate_report(results, exp_dir):
    """Generate comparison report."""
    lines = [
        "# Layer Mixing Ablation - Comparison Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        "\n## Summary Table\n",
        "| Pair | Age | Variant | WH Ratio | Jaw Width | ArcFace Father | ArcFace Mother |",
        "|------|-----|---------|----------|-----------|----------------|----------------|"
    ]
    
    for r in results:
        if 'wh_ratio_mean' in r:
            lines.append(f"| {r['pair']} | {r['age']} | {r['variant']} | "
                        f"{r.get('wh_ratio_mean', 'N/A'):.4f} | "
                        f"{r.get('norm_jaw_width_mean', 'N/A'):.4f} | "
                        f"{r.get('arcface_father_mean', 'N/A'):.4f} | "
                        f"{r.get('arcface_mother_mean', 'N/A'):.4f} |")
    
    lines.append("\n## Improvement vs Baseline (50/50)\n")
    lines.append("| Pair | Age | Variant | ΔWH Ratio | ΔJaw Width | ΔArcFace F | ΔArcFace M |")
    lines.append("|------|-----|---------|-----------|------------|------------|------------|")
    
    # Compute deltas
    for pair_name in set(r['pair'] for r in results):
        for age in ['5-10', '11-15', '16-21']:
            baseline = next((r for r in results if r['pair']==pair_name and r['age']==age and r['variant']=='baseline_50_50'), None)
            if not baseline:
                continue
            for variant in ['male_bias_0.6', 'male_bias_0.7', 'male_bias_0.8']:
                var = next((r for r in results if r['pair']==pair_name and r['age']==age and r['variant']==variant), None)
                if not var:
                    continue
                for metric in ['wh_ratio', 'norm_jaw_width', 'arcface_father', 'arcface_mother']:
                    pass  # compute deltas
    
    report_path = exp_dir / 'comparison_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    from datetime import datetime
    results = main()