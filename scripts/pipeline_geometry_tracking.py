"""
Pipeline Geometry Tracking
Tracks facial geometry at every stage of the KinshipForge pipeline.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from datetime import datetime
from pathlib import Path
from scipy import stats
import tempfile
import random
import csv

sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegene.api import init_model, tensor2rgb, load_img
from models.stylegene.gene_crossover_mutation import fuse_latent, REGION_SENSITIVITY_MAP, face_class, reparameterize
from models.stylegene.gene_pool import GenePoolFactory
from models.stylegene.fair_face_model import init_fair_model, predict_race
from preprocess.align_images import align_face
from models.stylegene.util import load_img
from configs import path_ckpt_genepool, path_ckpt_fairface
from scripts.legacy.geometry_utils import GeometryEstimator

DEVICE = torch.device("cpu")
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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/w2sub_sub2w_invertibility/pipeline_tracking')
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'intermediate_images').mkdir(exist_ok=True)
(OUTPUT_DIR / 'intermediate_latents').mkdir(exist_ok=True)

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_geometry(img):
    geom = geom_estimator.estimate_image_geometry(img)
    if geom is None:
        return {k: -1 for k in ['wh_ratio', 'jaw_width', 'cheek_width', 'temple_width', 'face_contour_area']}
    return {
        'wh_ratio': geom.get('Width/Height Ratio', -1),
        'jaw_width': geom.get('Jaw Width', -1),
        'cheek_width': geom.get('Cheek Width', -1),
        'temple_width': geom.get('Temple Width', -1),
        'face_contour_area': geom.get('Face Contour Area', -1),
    }

def load_and_encode(image_path, encoder, mean_latent, device):
    raw = cv2.imread(image_path)
    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    aligned = align_face(raw_rgb)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    img_t = load_img(tmp.name).to(device)
    os.unlink(tmp.name)
    with torch.no_grad():
        w18 = encoder(F.interpolate(img_t, size=(256, 256))) + mean_latent
    return w18, aligned

def save_image(img_tensor, path):
    img_np = tensor2rgb(img_tensor)
    Image.fromarray(img_np).save(path)

def latent_stats(w18):
    with torch.no_grad():
        w_flat = w18.squeeze(0)  # [18, 512]
        norms = torch.norm(w_flat, p=2, dim=-1)
        mean_norm = float(norms.mean().item())
        std_norm = float(norms.std().item())
        mean_vec = w_flat.mean(dim=0)
        centered = w_flat - mean_vec
        cov = centered.T @ centered / 17
        trace = float(torch.trace(cov).item())
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        eff_rank = float((eigvals.sum() ** 2 / (eigvals ** 2).sum()).item()) if len(eigvals) > 0 else 0.0
    return {
        'mean_norm': mean_norm, 'std_norm': std_norm,
        'cov_trace': trace, 'effective_rank': eff_rank,
        'eigenvalues': eigvals.tolist()
    }

def run_pipeline_tracking():
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
    
    all_results = []
    
    for f_file, m_file, gender, race_f, race_m, pair_name in TEST_PAIRS:
        print(f"\n{'='*60}")
        print(f"Processing {pair_name}...")
        print(f"{'='*60}")
        
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        # Stage 0: Original images
        w18_F, aligned_F = load_and_encode(f_path, encoder, mean_latent, DEVICE)
        w18_M, aligned_M = load_and_encode(m_path, encoder, mean_latent, DEVICE)
        
        geom_F_orig = compute_geometry(aligned_F)
        geom_M_orig = compute_geometry(aligned_M)
        
        # Stage 1: e4e inversion -> W+ reconstruction
        with torch.no_grad():
            img_F_w, _ = generator([w18_F], return_latents=True, input_is_latent=True)
            img_M_w, _ = generator([w18_M], return_latents=True, input_is_latent=True)
        geom_F_w = compute_geometry(tensor2rgb(img_F_w))
        geom_M_w = compute_geometry(tensor2rgb(img_M_w))
        
        save_image(img_F_w, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_father_Wplus.png")
        save_image(img_M_w, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_mother_Wplus.png")
        
        # Stage 2: W2Sub decomposition
        with torch.no_grad():
            mu_F, var_F, sub34_F = w2sub34(w18_F)
            mu_M, var_M, sub34_M = w2sub34(w18_M)
        
        # Stage 2b: Sub2W roundtrip (no crossover)
        with torch.no_grad():
            w18_F_rt = sub2w(sub34_F)
            w18_M_rt = sub2w(sub34_M)
            img_F_rt, _ = generator([w18_F_rt], return_latents=True, input_is_latent=True)
            img_M_rt, _ = generator([w18_M_rt], return_latents=True, input_is_latent=True)
        geom_F_rt = compute_geometry(tensor2rgb(img_F_rt))
        geom_M_rt = compute_geometry(tensor2rgb(img_M_rt))
        
        save_image(img_F_rt, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_father_roundtrip.png")
        save_image(img_M_rt, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_mother_roundtrip.png")
        
        # Stage 3: ARCS gamma computation
        s_map = REGION_SENSITIVITY_MAP
        s_vals = list(s_map.values())
        s_min = min(s_vals) if s_vals else 0.0
        s_max = max(s_vals) if s_vals else 1.0
        s_range = s_max - s_min if s_max != s_min else 1.0
        
        arcs_gammas = {}
        for name in face_class:
            if name == 'background':
                continue
            s_val = s_map.get(name, 0.0)
            s_norm = (s_val - s_min) / s_range
            # Using gamma=0.05, arcs_lambda=0.0 as default
            g_val = 0.05 * (1.0 - 0.0 * s_norm)
            arcs_gammas[name] = g_val
        
        # Stage 4: BRDAS sampling
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
        
        # Stage 5: Full pipeline (crossover + mutation + mix)
        w18_child = fuse_latent(
            w2sub34, sub2w, w18_F=w18_F, w18_M=w18_M,
            random_fakes=random_fakes, fixed_gamma=0.05, fixed_eta=0.4, arcs_lambda=0.0,
            child_gender=gender, geometry_weight=0.7, texture_weight=0.5
        )
        
        with torch.no_grad():
            img_child, _ = generator([w18_child], return_latents=True, input_is_latent=True)
        geom_child = compute_geometry(tensor2rgb(img_child))
        
        save_image(img_child, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_child_final.png")
        
        # Stage 6: Detailed crossover tracking (without mix)
        # Track regional latents through crossover
        with torch.no_grad():
            mu_F, var_F, sub34_F = w2sub34(w18_F)
            mu_M, var_M, sub34_M = w2sub34(w18_M)
        
        # Recreate crossover step-by-step
        s_map = REGION_SENSITIVITY_MAP
        s_vals = list(s_map.values())
        s_min = min(s_vals) if s_vals else 0.0
        s_max = max(s_vals) if s_vals else 1.0
        s_range = s_max - s_min if s_max != s_min else 1.0
        
        resolved_gammas = {}
        for name in face_class:
            if name == 'background':
                continue
            s_val = s_map.get(name, 0.0)
            s_norm = (s_val - s_min) / s_range
            g_val = 0.05 * (1.0 - 0.0 * s_norm)
            resolved_gammas[name] = g_val
        
        weights = {}
        for name in face_class:
            g_val = resolved_gammas.get(name, 0.05)
            weights[name] = (np.random.uniform(0, 1 - g_val), g_val)
        
        cur_class = np.random.choice(face_class, int(len(face_class) * (1 - 0.4)), replace=False).tolist()
        
        new_sub34 = torch.zeros_like(sub34_F, dtype=torch.float, device=DEVICE)
        
        # Track crossover per region
        crossover_region_stats = {}
        for i, classname in enumerate(face_class):
            if classname == 'background':
                new_sub34[:, :, i, :] = reparameterize(mu_F[:, :, i, :], var_F[:, :, i, :])
                continue
            
            fake_mu, fake_var = random.choice(random_fakes)
            w_i, b_i = weights[classname]
            
            if classname in cur_class:
                new_sub34[:, :, i, :] = reparameterize(
                    mu_F[:, :, i, :] * w_i + fake_mu[:, :, i, :].to(DEVICE) * b_i + mu_M[:, :, i, :] * (1 - w_i - b_i),
                    var_F[:, :, i, :] * w_i + fake_var[:, :, i, :].to(DEVICE) * b_i + var_M[:, :, i, :] * (1 - w_i - b_i))
            else:
                fake_latent = reparameterize(fake_mu[:, :, i, :], fake_var[:, :, i, :]).to(DEVICE)
                new_sub34[:, :, i, :] = new_sub34[:, :, i, :] + fake_latent
            
            # Track displacement for this region
            disp_from_F = torch.norm(new_sub34[:, :, i, :] - mu_F[:, :, i, :], p=2).item()
            disp_from_M = torch.norm(new_sub34[:, :, i, :] - mu_M[:, :, i, :], p=2).item()
            disp_from_fake = torch.norm(new_sub34[:, :, i, :] - fake_mu[:, :, i, :].to(DEVICE), p=2).item()
            
            crossover_region_stats[classname] = {
                'gamma': resolved_gammas.get(classname, 0),
                'weight_father': w_i,
                'weight_fake': b_i,
                'weight_mother': 1 - w_i - b_i,
                'in_crossover_class': classname in cur_class,
                'disp_from_father': disp_from_F,
                'disp_from_mother': disp_from_M,
                'disp_from_fake': disp_from_fake,
                'father_norm': torch.norm(mu_F[:, :, i, :], p=2).item(),
                'mother_norm': torch.norm(mu_M[:, :, i, :], p=2).item(),
                'fake_norm': torch.norm(fake_mu[:, :, i, :], p=2).item(),
            }
        
        with torch.no_grad():
            w18_after_crossover = sub2w(new_sub34)
            img_after_crossover, _ = generator([w18_after_crossover], return_latents=True, input_is_latent=True)
        geom_after_crossover = compute_geometry(tensor2rgb(img_after_crossover))
        
        save_image(img_after_crossover, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_after_crossover.png")
        
        # Apply mix
        from models.stylegene.gene_crossover_mutation import mix
        w18_final = mix(w18_F, w18_M, w18_after_crossover, 
                        geometry_weight=0.7, texture_weight=0.5, child_gender=gender)
        
        with torch.no_grad():
            img_final, _ = generator([w18_final], return_latents=True, input_is_latent=True)
        geom_final = compute_geometry(tensor2rgb(img_final))
        
        # Compute Mahalanobis distances
        def mahal_dist(w, mean_w):
            diff = w - mean_w
            return float(torch.norm(diff, p=2).item())
        
        # Aggregate results
        result = {
            'pair': pair_name,
            'timestamp': datetime.now().isoformat(),
            'stages': {
                'stage0_original': {
                    'father': geom_F_orig,
                    'mother': geom_M_orig,
                },
                'stage1_e4e_inversion': {
                    'father': geom_F_w,
                    'mother': geom_M_w,
                    'delta_father': {k: geom_F_w[k] - geom_F_orig[k] for k in geom_F_orig},
                    'delta_mother': {k: geom_M_w[k] - geom_M_orig[k] for k in geom_M_orig},
                },
                'stage2_w2sub_sub2w_roundtrip': {
                    'father': geom_F_rt,
                    'mother': geom_M_rt,
                    'delta_father': {k: geom_F_rt[k] - geom_F_w[k] for k in geom_F_w},
                    'delta_mother': {k: geom_M_rt[k] - geom_M_w[k] for k in geom_M_w},
                },
                'stage3_crossover': {
                    'geometry': geom_after_crossover,
                    'delta_from_wplus': {k: geom_after_crossover[k] - geom_F_w[k] for k in geom_F_w},
                    'delta_from_roundtrip': {k: geom_after_crossover[k] - geom_F_rt[k] for k in geom_F_rt},
                },
                'stage4_mix': {
                    'geometry': geom_final,
                    'delta_from_crossover': {k: geom_final[k] - geom_after_crossover[k] for k in geom_after_crossover},
                    'delta_from_wplus': {k: geom_final[k] - geom_F_w[k] for k in geom_F_w},
                },
            },
            'latent_statistics': {
                'father_original': latent_stats(w18_F),
                'father_roundtrip': latent_stats(w18_F_rt),
                'child_final': latent_stats(w18_child),
            },
            'mahalanobis_distances': {
                'father_to_mean': mahal_dist(w18_F, mean_latent),
                'father_rt_to_mean': mahal_dist(w18_F_rt, mean_latent),
                'child_to_mean': mahal_dist(w18_child, mean_latent),
                'child_to_father': mahal_dist(w18_child, w18_F),
            },
            'crossover_region_stats': crossover_region_stats,
            'arcs_gammas': arcs_gammas,
            'brdas_ancestry': None,  # Would need BRDAS logging
        }
        all_results.append(result)
        
        # Save per-pair
        with open(OUTPUT_DIR / 'intermediate_latents' / f"{pair_name}_full_pipeline.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        # Print summary
        print(f"  Father: orig WH={geom_F_orig['wh_ratio']:.4f} -> W+={geom_F_w['wh_ratio']:.4f} -> RT={geom_F_rt['wh_ratio']:.4f} -> Cross={geom_after_crossover['wh_ratio']:.4f} -> Final={geom_final['wh_ratio']:.4f}")
        print(f"  Mother: orig WH={geom_M_orig['wh_ratio']:.4f} -> W+={geom_M_w['wh_ratio']:.4f} -> RT={geom_M_rt['wh_ratio']:.4f}")
        print(f"  Child:  Final WH={geom_final['wh_ratio']:.4f}")
        print(f"  Mahal(F->mean): {mahal_dist(w18_F, mean_latent):.2f} -> Child: {mahal_dist(w18_child, mean_latent):.2f}")
    
    # Aggregate analysis
    print("\n" + "="*60)
    print("AGGREGATE PIPELINE GEOMETRY TRACKING")
    print("="*60)
    
    metrics = ['wh_ratio', 'jaw_width', 'cheek_width']
    
    # Stage 1: e4e inversion
    print("\nStage 1: e4e Inversion (RGB -> W+)")
    for role, label in [('father', 'Father'), ('mother', 'Mother')]:
        for metric in metrics:
            vals = []
            for r in all_results:
                orig = r['stages']['stage0_original'][role].get(metric, -1)
                wplus = r['stages']['stage1_e4e_inversion'][role].get(metric, -1)
                if orig > 0 and wplus > 0:
                    vals.append(wplus - orig)
            if vals:
                print(f"  {label}: d={np.mean(vals):+.4f} ± {np.std(vals):.4f}")
    
    # Stage 2: W2Sub+Sub2W roundtrip
    print("\nStage 2: W2Sub + Sub2W Roundtrip (W+ -> Regional -> W+)")
    for role, label in [('father', 'Father'), ('mother', 'Mother')]:
        for metric in metrics:
            vals = []
            for r in all_results:
                wplus = r['stages']['stage1_e4e_inversion'][role].get(metric, -1)
                rt = r['stages']['stage2_w2sub_sub2w_roundtrip'][role].get(metric, -1)
                if wplus > 0 and rt > 0:
                    vals.append(rt - wplus)
            if vals:
                print(f"  {label}: d={np.mean(vals):+.4f} ± {np.std(vals):.4f}")
    
    # Stage 3: Crossover (Father only)
    print("\nStage 3: Crossover (Father W+ -> Child W+)")
    for metric in metrics:
        vals = []
        for r in all_results:
            wplus = r['stages']['stage1_e4e_inversion']['father'].get(metric, -1)
            cross = r['stages']['stage3_crossover'].get('father_geometry', {}).get(metric, -1)
            if wplus > 0 and cross > 0:
                vals.append(cross - wplus)
        if vals:
            print(f"  {metric}: Δ={np.mean(vals):+.4f} ± {np.std(vals):.4f}")
    
    # Stage 4: Mix (Child final vs after crossover)
    print("\nStage 4: Mix (After Crossover -> Final Child)")
    for metric in metrics:
        vals = []
        for r in all_results:
            cross = r['stages']['stage3_crossover']['geometry'].get(metric, -1)
            final = r['stages']['stage4_mix']['geometry'].get(metric, -1)
            if cross > 0 and final > 0:
                vals.append(final - cross)
        if vals:
            print(f"  {metric}: Δ={np.mean(vals):+.4f} ± {np.std(vals):.4f}")
    
    # Total pipeline (Father original -> Child final)
    print("\nTotal Pipeline (Father Original -> Child Final)")
    for metric in metrics:
        vals = []
        for r in all_results:
            orig = r['stages']['stage0_original']['father'].get(metric, -1)
            final = r['stages']['stage4_mix']['geometry'].get(metric, -1)
            if orig > 0 and final > 0:
                vals.append(final - orig)
        if vals:
            print(f"  {metric}: Δ={np.mean(vals):+.4f} ± {np.std(vals):.4f}")
    
    # Mahalanobis drift
    print("\nMahalanobis Distance to FFHQ Mean:")
    for label, key in [('Father', 'father_to_mean'), ('Father RT', 'father_rt_to_mean'), ('Child', 'child_to_mean')]:
        vals = [r['mahalanobis_distances'][key] for r in all_results]
        print(f"  {label}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")
    
    # Save aggregate
    with open(OUTPUT_DIR / 'pipeline_geometry_tracking.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_pairs': len(TEST_PAIRS),
            'all_results': all_results,
        }, f, indent=2)
    
    # CSV summary
    import csv
    with open(OUTPUT_DIR / 'geometry.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pair', 'stage', 'role', 'wh_ratio', 'jaw_width', 'cheek_width', 'temple_width', 'mahal_to_mean'])
        for r in all_results:
            for stage_name, stage_data in r['stages'].items():
                if 'father' in stage_data:
                    writer.writerow([r['pair'], stage_name, 'father', 
                                     stage_data['father']['wh_ratio'],
                                     stage_data['father']['jaw_width'],
                                     stage_data['father']['cheek_width'],
                                     stage_data['father']['temple_width'],
                                     r['mahalanobis_distances'].get('father_to_mean' if 'father' in str(stage_name) else 'child_to_mean', -1)])
                if 'mother' in stage_data:
                    writer.writerow([r['pair'], stage_name, 'mother',
                                     stage_data['mother']['wh_ratio'],
                                     stage_data['mother']['jaw_width'],
                                     stage_data['mother']['cheek_width'],
                                     stage_data['mother']['temple_width'],
                                     -1])
            # Child final
            writer.writerow([r['pair'], 'stage4_mix', 'child',
                             r['stages']['stage4_mix']['geometry']['wh_ratio'],
                             r['stages']['stage4_mix']['geometry']['jaw_width'],
                             r['stages']['stage4_mix']['geometry']['cheek_width'],
                             r['stages']['stage4_mix']['geometry']['temple_width'],
                             r['mahalanobis_distances']['child_to_mean']])
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    return all_results

if __name__ == '__main__':
    run_pipeline_tracking()