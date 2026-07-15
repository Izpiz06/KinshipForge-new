"""
Stage 4: Mutation Analysis
Does mutation change shape or only texture?
Disable mutation (eta=0) vs Enable mutation (eta=0.4)
Compare: layer-wise displacement, geometry, identity
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from pathlib import Path
from scipy import stats
import tempfile

sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegene.api import init_model, tensor2rgb
from models.stylegene.gene_crossover_mutation import fuse_latent
from models.stylegene.gene_pool import GenePoolFactory
from models.stylegene.fair_face_model import init_fair_model, predict_race
from preprocess.align_images import align_face
from models.stylegene.util import load_img
from configs import path_ckpt_genepool, path_ckpt_fairface
from scripts.legacy.geometry_utils import GeometryEstimator

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
geom_estimator = GeometryEstimator()

PHOTOS = 'C:/Users/mdiza/coding/KinshipForge-iz/archive'
TEST_PAIRS = [
    ("father_p1.jpg", "mother_p1.jpg", "male", "Indian", "Indian", "P1_Shahrukh_Gauri"),
    ("father_p2.jpg", "mother_p2.jpeg", "male", "East Asian", "East Asian", "P2_Jackie_Joan"),
    ("father_p3.jpg", "mother_p3.jpeg", "male", "Black", "Black", "P3_Obama_Michelle"),
    ("father_p4.jpg", "mother_p4.jpg", "male", "White", "White", "P4_TomHanks_Rita"),
    ("father_p5.jpg", "mother_p5.jpg", "male", "Black", "White", "P5_Ben_Laura"),
]

POOL_AGE_MAP = {'5-10': '3-9', '11-15': '10-19', '16-21': '20-29'}
DISPLAY_AGE = '5-10'
POOL_AGE = POOL_AGE_MAP[DISPLAY_AGE]

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_geometry(img):
    geom = geom_estimator.estimate_image_geometry(img)
    if geom is None:
        return {k: -1 for k in ['wh_ratio', 'jaw_width', 'cheek_width', 'temple_width', 'face_contour_area', 'norm_jaw_width', 'norm_cheek_width']}
    return {
        'wh_ratio': geom.get('Width/Height Ratio', -1),
        'jaw_width': geom.get('Jaw Width', -1),
        'cheek_width': geom.get('Cheek Width', -1),
        'temple_width': geom.get('Temple Width', -1),
        'face_contour_area': geom.get('Face Contour Area', -1),
        'norm_jaw_width': geom.get('Jaw Width', -1),
        'norm_cheek_width': geom.get('Cheek Width', -1),
    }

def latent_stats(w18):
    with torch.no_grad():
        norms = torch.norm(w18, p=2, dim=-1)
        mean_norm = float(norms.mean().item())
        std_norm = float(norms.std().item())
        w_flat = w18.squeeze(0)
        mean_vec = w_flat.mean(dim=0)
        centered = w_flat - mean_vec
        cov = centered.T @ centered / (w_flat.shape[0] - 1)
        cov_trace = float(torch.trace(cov).item())
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        if len(eigvals) > 0:
            eff_rank = float((eigvals.sum() ** 2 / (eigvals ** 2).sum()).item())
            entropy = float(0.5 * (18 * np.log(2 * np.pi * np.e) + torch.log(eigvals).sum()).item())
        else:
            eff_rank = 0.0
            entropy = 0.0
    return {
        'mean_norm': mean_norm, 'std_norm': std_norm,
        'mean_vector': mean_vec.cpu().numpy().tolist(),
        'covariance_trace': cov_trace, 'effective_rank': eff_rank, 'entropy': entropy,
    }

def layer_wise_analysis(w18_ref, w18_test):
    """Per-layer displacement analysis."""
    layers = []
    for k in range(18):
        ref = w18_ref[:, k, :].flatten()
        test = w18_test[:, k, :].flatten()
        l2 = float(torch.norm(test - ref, p=2).item())
        cos = float(F.cosine_similarity(ref, test, dim=0).item())
        layers.append({'layer_idx': k, 'l2_displacement': l2, 'cosine_change': cos})
    return layers

def run_stage4():
    set_seed(42)
    print("Loading models...")
    encoder, generator, sub2w, w2sub34, mean_latent = init_model()
    encoder = encoder.to(DEVICE)
    generator = generator.to(DEVICE)
    sub2w = sub2w.to(DEVICE)
    w2sub34 = w2sub34.to(DEVICE)
    mean_latent = mean_latent.to(DEVICE)
    
    encoder.eval()
    generator.eval()
    sub2w.eval()
    w2sub34.eval()
    
    # Load gene pool
    pool_data = torch.load(path_ckpt_genepool, map_location='cpu', weights_only=False)
    geneFactor = GenePoolFactory(root_ffhq=None, device=DEVICE, mean_latent=mean_latent, max_sample=300)
    geneFactor.pools = pool_data
    
    model_fair_7 = init_fair_model(DEVICE)
    
    results = []
    
    for f_file, m_file, gender, race_f, race_m, pair_name in TEST_PAIRS:
        print(f"\nProcessing {pair_name}...")
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        # Load and encode both parents
        def load_encode(img_path):
            raw = cv2.imread(img_path)
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            aligned = align_face(raw_rgb)
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.close()
            cv2.imwrite(tmp.name, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
            img_t = load_img(tmp.name).to(DEVICE)
            os.unlink(tmp.name)
            with torch.no_grad():
                w18 = encoder(F.interpolate(img_t, size=(256, 256))) + mean_latent
            return w18, aligned
        
        w18_F, aligned_F = load_encode(f_path)
        w18_M, aligned_M = load_encode(m_path)
        
        # Get gene pools
        race_F_det, _, _, _ = predict_race(model_fair_7, 
            torch.from_numpy(aligned_F.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)/255.0, DEVICE)
        race_M_det, _, _, _ = predict_race(model_fair_7,
            torch.from_numpy(aligned_M.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)/255.0, DEVICE)
        
        pool_F = geneFactor(encoder, w2sub34(w18_F)[2], POOL_AGE, gender, race_F_det)
        pool_M = geneFactor(encoder, w2sub34(w18_M)[2], POOL_AGE, gender, race_M_det)
        
        if not pool_F:
            for age in ['0-2', '3-9', '10-19', '20-29']:
                if age != POOL_AGE:
                    pool_F += geneFactor(encoder, w2sub34(w18_F)[2], age, gender, race_F_det)
        if not pool_M:
            for age in ['0-2', '3-9', '10-19', '20-29']:
                if age != POOL_AGE:
                    pool_M += geneFactor(encoder, w2sub34(w18_M)[2], age, gender, race_M_det)
        
        from models.stylegene.api import brdas_sampler
        random_fakes = brdas_sampler(pool_F, pool_M, 0.5, 0.5)
        
        # BASELINE: W+ from just father (no crossover, no mutation, no mix)
        # Actually let's test the full pipeline with different eta values
        
        # Condition 1: No crossover, no mutation (baseline - just Sub2W of father)
        with torch.no_grad():
            _, _, sub34_F = w2sub34(w18_F)
            w18_baseline = sub2w(sub34_F)
        geom_baseline = compute_geometry(tensor2rgb(generator([w18_baseline], return_latents=True, input_is_latent=True)[0]))
        lat_baseline = latent_stats(w18_baseline)
        layer_baseline = layer_wise_analysis(w18_F, w18_baseline)
        
        # Condition 2: Crossover ONLY (eta=0, no mutation)
        w18_cross_only = fuse_latent(
            w2sub34, sub2w, w18_F=w18_F, w18_M=w18_M,
            random_fakes=random_fakes, fixed_gamma=0.05, fixed_eta=0.0, arcs_lambda=0.0,
            child_gender='male', geometry_weight=0.5, texture_weight=0.5
        )
        geom_cross_only = compute_geometry(tensor2rgb(generator([w18_cross_only], return_latents=True, input_is_latent=True)[0]))
        lat_cross_only = latent_stats(w18_cross_only)
        layer_cross_only = layer_wise_analysis(w18_F, w18_cross_only)
        
        # Condition 3: Crossover + Mutation (full, eta=0.4)
        w18_full = fuse_latent(
            w2sub34, sub2w, w18_F=w18_F, w18_M=w18_M,
            random_fakes=random_fakes, fixed_gamma=0.05, fixed_eta=0.4, arcs_lambda=0.0,
            child_gender='male', geometry_weight=0.5, texture_weight=0.5
        )
        geom_full = compute_geometry(tensor2rgb(generator([w18_full], return_latents=True, input_is_latent=True)[0]))
        lat_full = latent_stats(w18_full)
        layer_full = layer_wise_analysis(w18_F, w18_full)
        
        # Condition 4: Mutation ONLY (no crossover - use only father's sub34 + mutation)
        # This tests mutation in isolation
        with torch.no_grad():
            mu_F, var_F, sub34_F = w2sub34(w18_F)
        # Create fake pool from father's own stats for mutation-only test
        fake_pool = [(mu_F.cpu(), var_F.cpu())]
        w18_mut_only = fuse_latent(
            w2sub34, sub2w, w18_F=w18_F, w18_M=w18_F,  # Same parent
            random_fakes=fake_pool, fixed_gamma=0.05, fixed_eta=0.4, arcs_lambda=0.0,
            child_gender='male', geometry_weight=0.5, texture_weight=0.5
        )
        geom_mut_only = compute_geometry(tensor2rgb(generator([w18_mut_only], return_latents=True, input_is_latent=True)[0]))
        lat_mut_only = latent_stats(w18_mut_only)
        layer_mut_only = layer_wise_analysis(w18_F, w18_mut_only)
        
        # Results
        result = {
            'stage_name': 'mutation',
            'input_representation': '34 regional latents (post-crossover)',
            'output_representation': 'W+ latent after mutation',
            'pair': pair_name,
            'geometry_metrics': {
                'baseline': geom_baseline,
                'crossover_only': geom_cross_only,
                'full': geom_full,
                'mutation_only': geom_mut_only,
                'delta_cross': {k: geom_cross_only[k] - geom_baseline[k] for k in geom_baseline},
                'delta_full': {k: geom_full[k] - geom_baseline[k] for k in geom_baseline},
                'delta_mut_only': {k: geom_mut_only[k] - geom_baseline[k] for k in geom_baseline},
            },
            'identity_metrics': {
                'arcface_similarity_original': -1,
                'arcface_similarity_m_original': -1,
                'arcface_similarity_child': -1,
            },
            'latent_statistics': lat_full,
            'pca_analysis': None,
            'distortion_metrics': {
                'l2_distance_input_output': float(torch.norm(w18_full - w18_F, p=2).item()),
                'cosine_similarity_input_output': float(F.cosine_similarity(w18_full.flatten(), w18_F.flatten(), dim=0).item()),
                'mahalanobis_distance': float(torch.norm(w18_full - mean_latent, p=2).item()),
                'latent_norm_drift': lat_full['mean_norm'] - lat_baseline['mean_norm'],
                'variance_reduction_ratio': lat_full['covariance_trace'] / lat_baseline['covariance_trace'] if lat_baseline['covariance_trace'] > 0 else -1,
                'covariance_frobenius_change': -1,
            },
            'layer_wise_analysis': {
                'baseline': layer_baseline,
                'crossover_only': layer_cross_only,
                'full': layer_full,
                'mutation_only': layer_mut_only,
            },
            'region_wise_analysis': None,
            'visualizations_generated': [],
            'mathematical_findings': {
                'equation_analyzed': 'Mutation: new_sub34 = old_sub34 + var (where var = reparameterize(fake_mu, fake_var))',
                'bias_detected': False,
                'bias_description': 'Test if mutation changes shape (geometry layers 8-11) or only texture (12-17)',
                'proof_summary': 'Compare eta=0 vs eta=0.4 for geometry vs texture layer displacements',
            },
            'hypothesis_result': 'INCONCLUSIVE',
            'contribution_score': 0.0,
            'evidence_summary': '',
            'code_references': [
                'StyleGene/models/stylegene/gene_crossover_mutation.py:fuse_latent() lines 162-166 (mutation branch)',
            ],
        }
        results.append(result)
        
        print(f"  wh_ratio: base={geom_baseline['wh_ratio']:.4f}, cross={geom_cross_only['wh_ratio']:.4f}, full={geom_full['wh_ratio']:.4f}, mut={geom_mut_only['wh_ratio']:.4f}")
        print(f"  delta_cross={geom_cross_only['wh_ratio']-geom_baseline['wh_ratio']:.4f}, delta_full={geom_full['wh_ratio']-geom_baseline['wh_ratio']:.4f}, delta_mut={geom_mut_only['wh_ratio']-geom_baseline['wh_ratio']:.4f}")
        print(f"  jaw_width: base={geom_baseline['jaw_width']:.1f}, full={geom_full['jaw_width']:.1f}, delta={geom_full['jaw_width']-geom_baseline['jaw_width']:.1f}")
        print(f"  cheek_width: base={geom_baseline['cheek_width']:.1f}, full={geom_full['cheek_width']:.1f}, delta={geom_full['cheek_width']-geom_baseline['cheek_width']:.1f}")
    
    # Aggregate
    print("\n=== AGGREGATE STATISTICS ===")
    for metric in ['wh_ratio', 'jaw_width', 'cheek_width']:
        for label, delta_key in [('crossover', 'delta_cross'), ('full', 'delta_full'), ('mutation_only', 'delta_mut_only')]:
            deltas = [r['geometry_metrics'][delta_key][metric] for r in results if r['geometry_metrics'][delta_key][metric] != 0]
            if deltas:
                t_stat, p_val = stats.ttest_1samp(deltas, 0)
                d = np.mean(deltas) / np.std(deltas) if np.std(deltas) > 0 else 0
                ci = 1.96 * np.std(deltas) / np.sqrt(len(deltas))
                print(f"  {metric} ({label}): delta={np.mean(deltas):.4f}±{np.std(deltas):.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}, CI95=[{np.mean(deltas)-ci:.4f}, {np.mean(deltas)+ci:.4f}]")
                
                if label == 'mutation_only' and deltas:
                    if p_val < 0.05 and np.mean(deltas) > 0:
                        for r in results: r['hypothesis_result'] = 'CONFIRMED'
                    elif p_val < 0.05 and np.mean(deltas) < 0:
                        for r in results: r['hypothesis_result'] = 'FALSIFIED'
                    else:
                        for r in results: r['hypothesis_result'] = 'INCONCLUSIVE'
                    
                    if metric == 'wh_ratio':
                        effect = abs(np.mean(deltas) / np.std(deltas)) if np.std(deltas) > 0 else 0
                        for r in results:
                            r['contribution_score'] = float(min(effect / 2.0, 1.0))
                            r['evidence_summary'] = f"Mutation-only wh_ratio delta: {np.mean(deltas):.4f}±{np.std(deltas):.4f} (n={len(deltas)}), p={p_val:.4f}, Cohen's d={np.mean(deltas)/np.std(deltas) if np.std(deltas)>0 else 0:.3f}"
    
    # Layer-wise analysis
    print("\n=== LAYER-WISE DISPLACEMENT (mutation only) ===")
    for k in range(18):
        l2s = [r['layer_wise_analysis']['mutation_only'][k]['l2_displacement'] for r in results]
        coss = [r['layer_wise_analysis']['mutation_only'][k]['cosine_change'] for r in results]
        if l2s:
            print(f"  Layer {k}: L2={np.mean(l2s):.2f}±{np.std(l2s):.2f}, cos={np.mean(coss):.4f}±{np.std(coss):.4f}")
    
    # Save
    output_dir = Path('C:/Users/mdiza/coding/KinshipForge-iz/kinshipforge-pipeline-bottleneck/results')
    output_dir.mkdir(exist_ok=True)
    
    for r in results:
        filename = f"Stage_4_Mutation_{r['pair']}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(r, f, indent=2)
    
    with open(output_dir / "Stage_4_Mutation_AGGREGATE.json", 'w') as f:
        json.dump({'stage_name': 'mutation', 'summary': {'n_samples': len(results)}, 'all_results': results}, f, indent=2)
    
    print("\nStage 4 complete. Results saved.")
    return results

if __name__ == '__main__':
    run_stage4()