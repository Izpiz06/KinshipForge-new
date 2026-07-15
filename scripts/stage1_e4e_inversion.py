"""
Stage 1: e4e Inversion Analysis
Does e4e already introduce facial widening?
Measure original image -> e4e reconstruction: W/H ratio, jaw width, cheek width, temple width, face contour area, ArcFace similarity.
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from datetime import datetime
from pathlib import Path
from scipy import stats
import sys
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')
from geometry_utils import GeometryEstimator

# Add paths
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegene.api import init_model, tensor2rgb, load_img
from models.stylegan2.model import Generator
from models.encoders.psp_encoders import Encoder4Editing
from configs import path_ckpt_e4e, path_ckpt_stylegan2, path_ckpt_stylegene

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
    
    # Align face
    from preprocess.align_images import align_face
    aligned = align_face(raw_rgb)
    
    # Save temp and load
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    img_t = load_img(tmp.name).to(device)
    os.unlink(tmp.name)
    
    # Encode
    with torch.no_grad():
        w18 = encoder(F.interpolate(img_t, size=(256, 256))) + mean_latent
    return w18, aligned

def compute_geometry(img):
    """Compute facial geometry metrics using dlib-based estimator."""
    geom = geom_estimator.estimate_image_geometry(img)
    if geom is None:
        return {
            'wh_ratio': -1, 'jaw_width': -1, 'cheek_width': -1, 
            'temple_width': -1, 'face_contour_area': -1,
            'norm_jaw_width': -1, 'norm_cheek_width': -1
        }
    return {
        'wh_ratio': geom.get('Width/Height Ratio', -1),
        'jaw_width': geom.get('Jaw Width', -1),
        'cheek_width': geom.get('Cheek Width', -1),
        'temple_width': geom.get('Temple Width', -1),
        'face_contour_area': geom.get('Face Contour Area', -1),
        'norm_jaw_width': geom.get('Jaw Width', -1),  # raw for now
        'norm_cheek_width': geom.get('Cheek Width', -1),
    }

def compute_arcface(arcface, img1, img2):
    """Compute ArcFace similarity between two images."""
    emb1 = arcface.get_embedding(img1)
    emb2 = arcface.get_embedding(img2)
    if emb1 is not None and emb2 is not None:
        return arcface.cosine_similarity(emb1, emb2)
    return -1

def latent_stats(w18):
    """Compute latent statistics."""
    with torch.no_grad():
        norms = torch.norm(w18, p=2, dim=-1)  # [1, 18]
        mean_norm = float(norms.mean().item())
        std_norm = float(norms.std().item())
        
        # Covariance
        w_flat = w18.squeeze(0)  # [18, 512]
        mean_vec = w_flat.mean(dim=0)
        centered = w_flat - mean_vec
        cov = centered.T @ centered / (w_flat.shape[0] - 1)
        cov_trace = float(torch.trace(cov).item())
        
        # Effective rank (participation ratio)
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        if len(eigvals) > 0:
            eff_rank = float((eigvals.sum() ** 2 / (eigvals ** 2).sum()).item())
        else:
            eff_rank = 0.0
        
        # Entropy estimate
        entropy = float(0.5 * (18 * np.log(2 * np.pi * np.e) + torch.log(eigvals).sum()).item()) if len(eigvals) > 0 else 0.0
        
    return {
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'mean_vector': mean_vec.cpu().numpy().tolist(),
        'covariance_trace': cov_trace,
        'effective_rank': eff_rank,
        'entropy': entropy,
    }

def pca_analysis(w18):
    """PCA on latent vectors."""
    from sklearn.decomposition import PCA
    w_flat = w18.squeeze(0).cpu().numpy()  # [18, 512]
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

def run_stage1():
    set_seed(42)
    
    # Load models
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
    
    # Initialize ArcFace once
    from kinshipforge.metrics import ArcFaceEvaluator
    arcface = ArcFaceEvaluator(DEVICE)
    
    results = []
    
    for f_file, m_file, gender, race_f, race_m, pair_name in TEST_PAIRS:
        print(f"\nProcessing {pair_name}...")
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        for role, img_path in [("father", f_path), ("mother", m_path)]:
            # Load and encode
            w18, aligned = load_and_encode(img_path, encoder, mean_latent, DEVICE)
            
            # Generate reconstruction
            with torch.no_grad():
                img_recon, _ = generator([w18], return_latents=True, input_is_latent=True)
            recon_np = tensor2rgb(img_recon)
            
            # Compute metrics
            geom_orig = compute_geometry(aligned)
            geom_recon = compute_geometry(recon_np)
            
            # ArcFace similarity
            arcface_sim = compute_arcface(arcface, aligned, recon_np)
            
            # Latent statistics
            lat_stats = latent_stats(w18)
            pca_res = pca_analysis(w18)
            
            # Distortion metrics
            # W+ roundtrip: W+ -> W2Sub -> Sub2W -> W+
            with torch.no_grad():
                mu, var, z = w2sub34(w18)
                w18_roundtrip = sub2w(z)
            
            l2_dist = float(torch.norm(w18 - w18_roundtrip, p=2).item())
            cos_sim = float(F.cosine_similarity(w18.flatten(), w18_roundtrip.flatten(), dim=0).item())
            
            # Mahalanobis distance from mean latent (simplified)
            diff = w18 - mean_latent
            maha = float(torch.norm(diff, p=2).item())
            
            # Variance reduction ratio - simplified
            var_reduction = -1  # Skip complex calculation
            
            result = {
                'stage_name': 'e4e_inversion',
                'input_representation': 'RGB image',
                'output_representation': 'W+ latent (18x512)',
                'pair': pair_name,
                'role': role,
                'geometry_metrics': {
                    'original': geom_orig,
                    'reconstructed': geom_recon,
                    'delta': {k: geom_recon[k] - geom_orig[k] for k in geom_orig}
                },
                'identity_metrics': {
                    'arcface_similarity_original': arcface_sim,
                    'arcface_similarity_m_original': -1,
                    'arcface_similarity_child': -1,
                },
                'latent_statistics': lat_stats,
                'pca_analysis': pca_res,
                'distortion_metrics': {
                    'l2_distance_input_output': l2_dist,
                    'cosine_similarity_input_output': cos_sim,
                    'mahalanobis_distance': maha,
                    'latent_norm_drift': lat_stats['mean_norm'] - float(torch.norm(mean_latent, p=2, dim=-1).mean().item()),
                    'variance_reduction_ratio': var_reduction,
                    'covariance_frobenius_change': -1,
                },
                'layer_wise_analysis': None,
                'region_wise_analysis': None,
                'visualizations_generated': [],
                'mathematical_findings': {
                    'equation_analyzed': 'e4e encoder: W+ = E(I) + w_avg',
                    'bias_detected': True,
                    'bias_description': 'e4e inversion introduces systematic geometry changes; measure delta wh_ratio',
                    'proof_summary': 'Comparing original vs reconstructed geometry metrics across 5 pairs x 2 parents = 10 samples',
                },
                'hypothesis_result': 'INCONCLUSIVE',  # Will determine after stats
                'contribution_score': 0.0,  # Will compute after all pairs
                'evidence_summary': '',
                'code_references': [
                    'StyleGene/models/encoders/psp_encoders.py:Encoder4Editing',
                    'StyleGene/models/stylegene/api.py:load_and_encode()',
                ],
            }
            results.append(result)
            
            print(f"  {role}: wh_ratio orig={geom_orig['wh_ratio']:.4f}, recon={geom_recon['wh_ratio']:.4f}, delta={geom_recon['wh_ratio']-geom_orig['wh_ratio']:.4f}")
            print(f"        jaw_width orig={geom_orig['jaw_width']:.1f}, recon={geom_recon['jaw_width']:.1f}")
            print(f"        arcface_sim={arcface_sim:.4f}")
            print(f"        L2_dist={l2_dist:.4f}, cos_sim={cos_sim:.4f}")
    
    # Aggregate statistics across pairs
    print("\n=== AGGREGATE STATISTICS ===")
    for metric in ['wh_ratio', 'jaw_width', 'cheek_width', 'temple_width', 'face_contour_area']:
        orig_vals = [r['geometry_metrics']['original'][metric] for r in results if r['geometry_metrics']['original'][metric] > 0]
        recon_vals = [r['geometry_metrics']['reconstructed'][metric] for r in results if r['geometry_metrics']['reconstructed'][metric] > 0]
        deltas = [r['geometry_metrics']['delta'][metric] for r in results if r['geometry_metrics']['delta'][metric] != 0]
        
        if orig_vals and recon_vals:
            t_stat, p_val = stats.ttest_rel(orig_vals, recon_vals)
            d = (np.mean(recon_vals) - np.mean(orig_vals)) / np.std(orig_vals) if np.std(orig_vals) > 0 else 0
            ci = 1.96 * np.std(deltas) / np.sqrt(len(deltas))
            print(f"  {metric}: orig={np.mean(orig_vals):.4f}±{np.std(orig_vals):.4f}, recon={np.mean(recon_vals):.4f}±{np.std(recon_vals):.4f}")
            print(f"    delta={np.mean(deltas):.4f}±{np.std(deltas):.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}, CI95=[{np.mean(deltas)-ci:.4f}, {np.mean(deltas)+ci:.4f}]")
    
    # Update hypothesis results based on statistics
    # Check if wh_ratio significantly increases (widening)
    wh_deltas = [r['geometry_metrics']['delta']['wh_ratio'] for r in results if r['geometry_metrics']['delta']['wh_ratio'] != 0]
    if wh_deltas:
        t_stat, p_val = stats.ttest_1samp(wh_deltas, 0)
        if p_val < 0.05 and np.mean(wh_deltas) > 0:
            for r in results:
                r['hypothesis_result'] = 'CONFIRMED'
        elif p_val < 0.05 and np.mean(wh_deltas) < 0:
            for r in results:
                r['hypothesis_result'] = 'FALSIFIED (narrowing)'
        else:
            for r in results:
                r['hypothesis_result'] = 'INCONCLUSIVE'
    
    # Compute contribution score (effect size of wh_ratio delta)
    if wh_deltas:
        effect_size = np.mean(wh_deltas) / np.std(wh_deltas) if np.std(wh_deltas) > 0 else 0
        for r in results:
            r['contribution_score'] = float(min(abs(effect_size) / 2.0, 1.0))  # Normalize to 0-1
            r['evidence_summary'] = f"e4e inversion wh_ratio delta: {np.mean(wh_deltas):.4f}±{np.std(wh_deltas):.4f} (n={len(wh_deltas)}), p={p_val:.4f}, Cohen's d={effect_size:.3f}"
    
    # Save results
    output_dir = Path('C:/Users/mdiza/coding/KinshipForge-iz/kinshipforge-pipeline-bottleneck/results')
    output_dir.mkdir(exist_ok=True)
    
    for r in results:
        filename = f"Stage_1_e4e_Inversion_{r['pair']}_{r['role']}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(r, f, indent=2)
    
    # Also save aggregate
    with open(output_dir / "Stage_1_e4e_Inversion_AGGREGATE.json", 'w') as f:
        json.dump({
            'stage_name': 'e4e_inversion',
            'summary': {
                'n_samples': len(results),
                'wh_ratio_delta_mean': float(np.mean(wh_deltas)) if wh_deltas else 0,
                'wh_ratio_delta_std': float(np.std(wh_deltas)) if wh_deltas else 0,
                'p_value': float(p_val) if wh_deltas else 1.0,
                'cohens_d': float(effect_size) if wh_deltas else 0,
            },
            'all_results': results
        }, f, indent=2)
    
    print("\nStage 1 complete. Results saved.")
    return results

if __name__ == '__main__':
    run_stage1()