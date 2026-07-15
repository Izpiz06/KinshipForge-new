"""
Stage 2: W2Sub Decomposition Analysis
Does decomposing W+ (18x512) into 34 regional latent spaces distort geometry?
Investigate W2Sub projection: latent statistics, covariance, PCA, cosine similarity, L2 distance.
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

# Add paths
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegene.api import init_model, tensor2rgb, load_img
from models.stylegene.model import MappingW2Sub, MappingSub2W
from models.encoders.psp_encoders import Encoder4Editing
from preprocess.align_images import align_face
from configs import path_ckpt_e4e, path_ckpt_stylegan2, path_ckpt_stylegene
from geometry_utils import GeometryEstimator

# Test pairs
PHOTOS = 'C:/Users/mdiza/coding/KinshipForge-iz/archive'
TEST_PAIRS = [
    ("father_p1.jpg", "mother_p1.jpg", "male", "Indian", "Indian", "P1_Shahrukh_Gauri"),
    ("father_p2.jpg", "mother_p2.jpeg", "male", "East Asian", "East Asian", "P2_Jackie_Joan"),
    ("father_p3.jpg", "mother_p3.jpeg", "male", "Black", "Black", "P3_Obama_Michelle"),
    ("father_p4.jpg", "mother_p4.jpg", "male", "White", "White", "P4_TomHanks_Rita"),
    ("father_p5.jpg", "mother_p5.jpg", "male", "Black", "White", "P5_Ben_Laura"),
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
geom_estimator = GeometryEstimator()

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_and_encode(image_path, encoder, mean_latent, device):
    raw = cv2.imread(image_path)
    if raw is None:
        raise ValueError(f"Could not load {image_path}")
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

def region_wise_analysis(mu, var, z, face_class):
    """Analyze per-region statistics from W2Sub output."""
    regions = []
    for i, name in enumerate(face_class):
        if name == 'background':
            continue
        mu_r = mu[:, :, i, :].squeeze(0)  # [18, 512]
        var_r = var[:, :, i, :].squeeze(0)
        z_r = z[:, :, i, :].squeeze(0)
        
        mu_norm = float(torch.norm(mu_r, p=2).item() / 18)
        var_mean = float(var_r.mean().item())
        
        regions.append({
            'region_name': name,
            'region_idx': i,
            'mean_norm': mu_norm,
            'variance': var_mean,
            'displacement_from_father': -1,  # Will fill later
            'displacement_from_mother': -1,
            'covariance_trace': float(torch.trace(z_r.T @ z_r / 17).item()) if z_r.shape[0] > 1 else 0,
        })
    return regions

def run_stage2():
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
    
    # Import face_class
    from models.stylegene.data_util import face_class
    
    results = []
    
    for f_file, m_file, gender, race_f, race_m, pair_name in TEST_PAIRS:
        print(f"\nProcessing {pair_name}...")
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        for role, img_path in [("father", f_path), ("mother", m_path)]:
            # Load and encode to W+
            w18, aligned = load_and_encode(img_path, encoder, mean_latent, DEVICE)
            
            # Generate reconstruction from W+ (before W2Sub)
            with torch.no_grad():
                img_w_plus, _ = generator([w18], return_latents=True, input_is_latent=True)
            w_plus_np = tensor2rgb(img_w_plus)
            
            # W2Sub decomposition
            with torch.no_grad():
                mu, var, z = w2sub34(w18)
            
            # Sub2W reconstruction (roundtrip)
            with torch.no_grad():
                w18_roundtrip = sub2w(z)
                img_roundtrip, _ = generator([w18_roundtrip], return_latents=True, input_is_latent=True)
            roundtrip_np = tensor2rgb(img_roundtrip)
            
            # Compute metrics
            geom_w_plus = compute_geometry(w_plus_np)
            geom_roundtrip = compute_geometry(roundtrip_np)
            
            # Latent stats
            lat_stats_w = latent_stats(w18)
            lat_stats_roundtrip = latent_stats(w18_roundtrip)
            
            # PCA
            pca_w = pca_analysis(w18)
            pca_roundtrip = pca_analysis(w18_roundtrip)
            
            # Distortion metrics
            l2_dist = float(torch.norm(w18 - w18_roundtrip, p=2).item())
            cos_sim = float(F.cosine_similarity(w18.flatten(), w18_roundtrip.flatten(), dim=0).item())
            
            # Mahalanobis distance from mean latent
            diff = w18 - mean_latent
            maha = float(torch.norm(diff, p=2).item())
            
            # Latent norm drift
            norm_drift = lat_stats_roundtrip['mean_norm'] - lat_stats_w['mean_norm']
            
            # Variance reduction ratio
            var_reduction = lat_stats_roundtrip['covariance_trace'] / lat_stats_w['covariance_trace'] if lat_stats_w['covariance_trace'] > 0 else -1
            
            # Covariance Frobenius change
            w_flat = w18.squeeze(0)
            centered = w_flat - w_flat.mean(dim=0)
            cov_w = centered.T @ centered / 17
            
            rt_flat = w18_roundtrip.squeeze(0)
            centered_rt = rt_flat - rt_flat.mean(dim=0)
            cov_rt = centered_rt.T @ centered_rt / 17
            
            cov_frob_change = float(torch.norm(cov_w - cov_rt, p='fro').item())
            
            # Region-wise analysis
            region_stats = region_wise_analysis(mu, var, z, face_class)
            
            result = {
                'stage_name': 'W2Sub_decomposition',
                'input_representation': 'W+ latent (18x512)',
                'output_representation': '34 regional latents (mu, var, z each 18x512)',
                'pair': pair_name,
                'role': role,
                'geometry_metrics': {
                    'w_plus': geom_w_plus,
                    'roundtrip': geom_roundtrip,
                    'delta': {k: geom_roundtrip[k] - geom_w_plus[k] for k in geom_w_plus}
                },
                'identity_metrics': {
                    'arcface_similarity_original': -1,
                    'arcface_similarity_m_original': -1,
                    'arcface_similarity_child': -1,
                },
                'latent_statistics': lat_stats_w,
                'pca_analysis': pca_w,
                'distortion_metrics': {
                    'l2_distance_input_output': l2_dist,
                    'cosine_similarity_input_output': cos_sim,
                    'mahalanobis_distance': maha,
                    'latent_norm_drift': norm_drift,
                    'variance_reduction_ratio': var_reduction,
                    'covariance_frobenius_change': cov_frob_change,
                },
                'layer_wise_analysis': None,
                'region_wise_analysis': {
                    'region_stats': region_stats
                },
                'visualizations_generated': [],
                'mathematical_findings': {
                    'equation_analyzed': 'W2Sub: W+ (18x512) -> 34 regions x (mu, var, z) each 18x512',
                    'bias_detected': False,
                    'bias_description': 'Measure geometry delta between W+ and roundtrip reconstruction',
                    'proof_summary': 'Compare geometry metrics before/after W2Sub->Sub2W roundtrip across 5 pairs x 2 parents',
                },
                'hypothesis_result': 'INCONCLUSIVE',
                'contribution_score': 0.0,
                'evidence_summary': '',
                'code_references': [
                    'StyleGene/models/stylegene/model.py:MappingW2Sub.forward()',
                    'StyleGene/models/stylegene/model.py:MappingSub2W.forward()',
                    'StyleGene/models/stylegene/gene_crossover_mutation.py:fuse_latent() lines 105-106',
                ],
            }
            results.append(result)
            
            print(f"  {role}: wh_ratio W+={geom_w_plus['wh_ratio']:.4f}, RT={geom_roundtrip['wh_ratio']:.4f}, delta={geom_roundtrip['wh_ratio']-geom_w_plus['wh_ratio']:.4f}")
            print(f"        L2={l2_dist:.4f}, cos={cos_sim:.4f}, var_reduction={var_reduction:.4f}")
    
    # Aggregate statistics
    print("\n=== AGGREGATE STATISTICS ===")
    for metric in ['wh_ratio', 'jaw_width', 'cheek_width', 'temple_width', 'face_contour_area']:
        deltas = [r['geometry_metrics']['delta'][metric] for r in results if r['geometry_metrics']['delta'][metric] != 0]
        if deltas:
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            d = np.mean(deltas) / np.std(deltas) if np.std(deltas) > 0 else 0
            ci = 1.96 * np.std(deltas) / np.sqrt(len(deltas))
            print(f"  {metric}: delta={np.mean(deltas):.4f}±{np.std(deltas):.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}, CI95=[{np.mean(deltas)-ci:.4f}, {np.mean(deltas)+ci:.4f}]")
            
            if p_val < 0.05 and np.mean(deltas) > 0:
                for r in results:
                    r['hypothesis_result'] = 'CONFIRMED'
            elif p_val < 0.05 and np.mean(deltas) < 0:
                for r in results:
                    r['hypothesis_result'] = 'FALSIFIED (narrowing)'
            else:
                for r in results:
                    r['hypothesis_result'] = 'INCONCLUSIVE'
            
            if metric == 'wh_ratio' and deltas:
                effect = abs(np.mean(deltas) / np.std(deltas)) if np.std(deltas) > 0 else 0
                for r in results:
                    r['contribution_score'] = float(min(effect / 2.0, 1.0))
                    r['evidence_summary'] = f"W2Sub roundtrip wh_ratio delta: {np.mean(deltas):.4f}±{np.std(deltas):.4f} (n={len(deltas)}), p={p_val:.4f}, Cohen's d={np.mean(deltas)/np.std(deltas) if np.std(deltas)>0 else 0:.3f}"
    
    # Save results
    output_dir = Path('C:/Users/mdiza/coding/KinshipForge-iz/kinshipforge-pipeline-bottleneck/results')
    output_dir.mkdir(exist_ok=True)
    
    for r in results:
        filename = f"Stage_2_W2Sub_Decomposition_{r['pair']}_{r['role']}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(r, f, indent=2)
    
    with open(output_dir / "Stage_2_W2Sub_Decomposition_AGGREGATE.json", 'w') as f:
        json.dump({
            'stage_name': 'W2Sub_decomposition',
            'summary': {
                'n_samples': len(results),
            },
            'all_results': results
        }, f, indent=2)
    
    print("\nStage 2 complete. Results saved.")
    return results

if __name__ == '__main__':
    run_stage2()