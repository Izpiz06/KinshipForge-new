"""
Stage 3: Regional Crossover Analysis
Is the linear Gaussian interpolation mathematically biased?
Equation: child = father*w + fake*gamma + mother*(1-w-gamma)
Investigate: latent norm drift, covariance collapse, movement toward FFHQ mean, variance reduction, latent manifold contraction.
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

sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegene.api import init_model, tensor2rgb
from models.stylegene.gene_crossover_mutation import fuse_latent, reparameterize, REGION_SENSITIVITY_MAP, face_class
from models.stylegene.gene_pool import GenePoolFactory
from models.stylegene.fair_face_model import init_fair_model, predict_race
from preprocess.align_images import align_face
from models.stylegene.util import load_img
from configs import path_ckpt_genepool, path_ckpt_fairface, path_ckpt_landmark68
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
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'mean_vector': mean_vec.cpu().numpy().tolist(),
        'covariance_trace': cov_trace,
        'effective_rank': eff_rank,
        'entropy': entropy,
    }

def pca_analysis(w18):
    from sklearn.decomposition import PCA
    w_flat = w18.squeeze(0).cpu().numpy()
    pca = PCA(n_components=min(18, 512))
    pca.fit(w_flat)
    explained_var = pca.explained_variance_ratio_.tolist()
    n_95 = int(np.argmax(np.cumsum(explained_var) >= 0.95)) + 1
    n_99 = int(np.argmax(np.cumsum(explained_var) >= 0.99)) + 1
    pc1 = pca.components_[0].tolist()
    return {
        'explained_variance_ratio': explained_var,
        'n_components_95': n_95,
        'n_components_99': n_99,
        'pc1_direction': pc1,
    }

def region_wise_stats(mu, var, z):
    """Compute per-region statistics from W2Sub output."""
    stats = []
    for i, name in enumerate(face_class):
        if name == 'background':
            continue
        mu_r = mu[:, :, i, :].squeeze(0)  # [18, 512]
        var_r = var[:, :, i, :].squeeze(0)
        z_r = z[:, :, i, :].squeeze(0)
        
        mu_norm = float(torch.norm(mu_r, p=2, dim=-1).mean().item())
        var_val = float(var_r.mean().item())
        z_norm = float(torch.norm(z_r, p=2, dim=-1).mean().item())
        
        # Covariance trace for this region
        centered = z_r - z_r.mean(dim=0)
        cov_r = centered.T @ centered / 17
        cov_trace = float(torch.trace(cov_r).item())
        
        stats.append({
            'region_name': name,
            'region_idx': i,
            'mu_norm': mu_norm,
            'var_mean': var_val,
            'z_norm': z_norm,
            'covariance_trace': cov_trace,
        })
    return stats

def run_stage3():
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
        
        # Load and encode
        for role, img_path, race in [("father", f_path, race_f), ("mother", m_path, race_m)]:
            raw = cv2.imread(img_path)
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            aligned = align_face(raw_rgb)
            
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.close()
            cv2.imwrite(tmp.name, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
            img_t = load_img(tmp.name).to(DEVICE)
            os.unlink(tmp.name)
            
            with torch.no_grad():
                w18 = encoder(F.interpolate(img_t, size=(256, 256))) + mean_latent
            
            # Get gene pool
            race_det, _, _, _ = predict_race(model_fair_7, 
                torch.from_numpy(aligned.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)/255.0, DEVICE)
            
            pool = geneFactor(encoder, w2sub34(w18)[2], POOL_AGE, gender, race_det)
            if not pool:
                for age in ['0-2', '3-9', '10-19', '20-29']:
                    if age != POOL_AGE:
                        pool += geneFactor(encoder, w2sub34(w18)[2], age, gender, race_det)
            
            if isinstance(pool, dict):
                from models.stylegene.api import brdas_sampler
                random_fakes = brdas_sampler(pool["father_pool"], pool["mother_pool"], 0.5, 0.5)
            else:
                random_fakes = pool
            
            # === BASELINE: W+ -> W2Sub -> Sub2W -> W+ (no crossover) ===
            with torch.no_grad():
                mu_F, var_F, sub34_F = w2sub34(w18)
                w18_baseline = sub2w(sub34_F)
            
            geom_baseline = compute_geometry(tensor2rgb(generator([w18_baseline], return_latents=True, input_is_latent=True)[0]))
            lat_stats_baseline = latent_stats(w18_baseline)
            pca_baseline = pca_analysis(w18_baseline)
            
            # === CROSSOVER WITH MUTATION (full fuse_latent) ===
            w18_crossover = fuse_latent(
                w2sub34, sub2w, w18_F=w18, w18_M=w18,  # Use same parent for controlled test
                random_fakes=random_fakes,
                fixed_gamma=0.05, fixed_eta=0.4, arcs_lambda=0.0,
                child_gender='male', geometry_weight=0.5, texture_weight=0.5
            )
            
            geom_crossover = compute_geometry(tensor2rgb(generator([w18_crossover], return_latents=True, input_is_latent=True)[0]))
            lat_stats_crossover = latent_stats(w18_crossover)
            pca_crossover = pca_analysis(w18_crossover)
            
            # === CROSSOVER ONLY (eta=0, no mutation) ===
            w18_crossover_only = fuse_latent(
                w2sub34, sub2w, w18_F=w18, w18_M=w18,
                random_fakes=random_fakes,
                fixed_gamma=0.05, fixed_eta=0.0, arcs_lambda=0.0,
                child_gender='male', geometry_weight=0.5, texture_weight=0.5
            )
            
            geom_crossover_only = compute_geometry(tensor2rgb(generator([w18_crossover_only], return_latents=True, input_is_latent=True)[0]))
            lat_stats_crossover_only = latent_stats(w18_crossover_only)
            
            # === ANALYSIS: Displacement from father ===
            disp_crossover = w18_crossover - w18
            disp_crossover_only = w18_crossover_only - w18
            
            l2_disp_crossover = float(torch.norm(disp_crossover, p=2).item())
            cos_disp_crossover = float(F.cosine_similarity(w18_crossover.flatten(), w18.flatten(), dim=0).item())
            
            l2_disp_crossover_only = float(torch.norm(disp_crossover_only, p=2).item())
            cos_disp_crossover_only = float(F.cosine_similarity(w18_crossover_only.flatten(), w18.flatten(), dim=0).item())
            
            # Norm drift
            norm_father = float(torch.norm(w18, p=2).item())
            norm_crossover = float(torch.norm(w18_crossover, p=2).item())
            norm_drift = norm_crossover - norm_father
            
            # Movement toward FFHQ mean
            diff_to_mean_crossover = w18_crossover - mean_latent
            diff_to_mean_father = w18 - mean_latent
            maha_crossover = float(torch.norm(diff_to_mean_crossover, p=2).item())
            maha_father = float(torch.norm(diff_to_mean_father, p=2).item())
            movement_toward_mean = maha_father - maha_crossover
            
            # Variance reduction
            var_reduction = lat_stats_crossover['covariance_trace'] / lat_stats_baseline['covariance_trace'] if lat_stats_baseline['covariance_trace'] > 0 else -1
            var_reduction_only = lat_stats_crossover_only['covariance_trace'] / lat_stats_baseline['covariance_trace'] if lat_stats_baseline['covariance_trace'] > 0 else -1
            
            # Region-wise analysis for crossover
            mu_F, var_F, sub34_F = w2sub34(w18)
            mu_M, var_M, sub34_M = w2sub34(w18)  # Same for controlled test
            
            # Get region stats from crossover
            mu_cross, var_cross, z_cross = w2sub34(w18_crossover)
            region_stats_cross = region_wise_stats(mu_cross, var_cross, z_cross)
            
            # Region stats from father
            region_stats_father = region_wise_stats(mu_F, var_F, sub34_F)
            
            # Per-region displacement
            region_displacements = []
            for rs_f, rs_c in zip(region_stats_father, region_stats_cross):
                region_displacements.append({
                    'region_name': rs_f['region_name'],
                    'mu_norm_father': rs_f['mu_norm'],
                    'mu_norm_crossover': rs_c['mu_norm'],
                    'mu_norm_delta': rs_c['mu_norm'] - rs_f['mu_norm'],
                    'z_norm_father': rs_f['z_norm'],
                    'z_norm_crossover': rs_c['z_norm'],
                    'z_norm_delta': rs_c['z_norm'] - rs_f['z_norm'],
                    'cov_trace_father': rs_f['covariance_trace'],
                    'cov_trace_crossover': rs_c['covariance_trace'],
                    'cov_trace_delta': rs_c['covariance_trace'] - rs_f['covariance_trace'],
                })
            
            result = {
                'stage_name': 'regional_crossover',
                'input_representation': '34 regional latents (father, mother, gene_pool)',
                'output_representation': 'W+ latent after crossover',
                'pair': pair_name,
                'role': role,
                'geometry_metrics': {
                    'baseline': geom_baseline,
                    'crossover_full': geom_crossover,
                    'crossover_only': geom_crossover_only,
                    'delta_full': {k: geom_crossover[k] - geom_baseline[k] for k in geom_baseline},
                    'delta_crossover_only': {k: geom_crossover_only[k] - geom_baseline[k] for k in geom_baseline},
                },
                'identity_metrics': {
                    'arcface_similarity_original': -1,
                    'arcface_similarity_m_original': -1,
                    'arcface_similarity_child': -1,
                },
                'latent_statistics': lat_stats_crossover,
                'pca_analysis': pca_crossover,
                'distortion_metrics': {
                    'l2_distance_input_output': l2_disp_crossover,
                    'cosine_similarity_input_output': cos_disp_crossover,
                    'mahalanobis_distance': maha_crossover,
                    'latent_norm_drift': norm_drift,
                    'variance_reduction_ratio': var_reduction,
                    'covariance_frobenius_change': -1,
                },
                'layer_wise_analysis': None,
                'region_wise_analysis': {
                    'region_stats': region_stats_cross,
                    'region_displacements': region_displacements,
                },
                'visualizations_generated': [],
                'mathematical_findings': {
                    'equation_analyzed': 'child = father*w + fake*gamma + mother*(1-w-gamma) per region',
                    'bias_detected': False,
                    'bias_description': 'Measure latent norm drift, covariance collapse, movement toward FFHQ mean, variance reduction',
                    'proof_summary': 'Compare baseline (no crossover) vs crossover-only (eta=0) vs full (eta=0.4) across 5 pairs',
                },
                'hypothesis_result': 'INCONCLUSIVE',
                'contribution_score': 0.0,
                'evidence_summary': '',
                'code_references': [
                    'StyleGene/models/stylegene/gene_crossover_mutation.py:fuse_latent() lines 151-166',
                    'StyleGene/models/stylegene/gene_crossover_mutation.py:reparameterize() line 45-51',
                ],
            }
            results.append(result)
            
            print(f"  {role}: wh_ratio baseline={geom_baseline['wh_ratio']:.4f}, cross={geom_crossover['wh_ratio']:.4f}, cross_only={geom_crossover_only['wh_ratio']:.4f}")
            print(f"        delta_full={geom_crossover['wh_ratio']-geom_baseline['wh_ratio']:.4f}, delta_cross_only={geom_crossover_only['wh_ratio']-geom_baseline['wh_ratio']:.4f}")
            print(f"        L2_disp_cross={l2_disp_crossover:.2f}, cos={cos_disp_crossover:.4f}, norm_drift={norm_drift:.2f}")
            print(f"        movement_to_mean={movement_toward_mean:.2f}, var_reduction={var_reduction:.4f}")
    
    # Aggregate
    print("\n=== AGGREGATE STATISTICS ===")
    for metric in ['wh_ratio', 'jaw_width', 'cheek_width']:
        deltas_full = [r['geometry_metrics']['delta_full'][metric] for r in results if r['geometry_metrics']['delta_full'][metric] != 0]
        deltas_cross_only = [r['geometry_metrics']['delta_crossover_only'][metric] for r in results if r['geometry_metrics']['delta_crossover_only'][metric] != 0]
        
        for label, deltas in [('full', deltas_full), ('crossover_only', deltas_cross_only)]:
            if deltas:
                t_stat, p_val = stats.ttest_1samp(deltas, 0)
                d = np.mean(deltas) / np.std(deltas) if np.std(deltas) > 0 else 0
                ci = 1.96 * np.std(deltas) / np.sqrt(len(deltas))
                print(f"  {metric} ({label}): delta={np.mean(deltas):.4f}±{np.std(deltas):.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}, CI95=[{np.mean(deltas)-ci:.4f}, {np.mean(deltas)+ci:.4f}]")
                
                if label == 'crossover_only' and deltas_cross_only:
                    if p_val < 0.05 and np.mean(deltas) > 0:
                        for r in results:
                            r['hypothesis_result'] = 'CONFIRMED'
                    elif p_val < 0.05 and np.mean(deltas) < 0:
                        for r in results:
                            r['hypothesis_result'] = 'FALSIFIED'
                    else:
                        for r in results:
                            r['hypothesis_result'] = 'INCONCLUSIVE'
                    
                    if metric == 'wh_ratio':
                        effect = abs(np.mean(deltas) / np.std(deltas)) if np.std(deltas) > 0 else 0
                        for r in results:
                            r['contribution_score'] = float(min(effect / 2.0, 1.0))
                            r['evidence_summary'] = f"Crossover-only wh_ratio delta: {np.mean(deltas):.4f}±{np.std(deltas):.4f} (n={len(deltas)}), p={p_val:.4f}, Cohen's d={np.mean(deltas)/np.std(deltas) if np.std(deltas)>0 else 0:.3f}"
    
    # Save
    output_dir = Path('C:/Users/mdiza/coding/KinshipForge-iz/kinshipforge-pipeline-bottleneck/results')
    output_dir.mkdir(exist_ok=True)
    
    for r in results:
        filename = f"Stage_3_Regional_Crossover_{r['pair']}_{r['role']}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(r, f, indent=2)
    
    with open(output_dir / "Stage_3_Regional_Crossover_AGGREGATE.json", 'w') as f:
        json.dump({
            'stage_name': 'regional_crossover',
            'summary': {'n_samples': len(results)},
            'all_results': results
        }, f, indent=2)
    
    print("\nStage 3 complete. Results saved.")
    return results

if __name__ == '__main__':
    run_stage3()