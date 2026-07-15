"""
Task 6: Representation Analysis
Determine whether Sub34 actually contains more information than W+,
or whether it is simply a redundant projection.

Compute:
- Effective rank
- Intrinsic dimensionality  
- PCA
- Entropy
- Variance explained
- Null space analysis
- Mutual information between regions
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

sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegene.api import init_model, tensor2rgb, load_img
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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/w2sub_sub2w_invertibility/representation_analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def compute_representation_metrics(w18, mu, var, z, sub34):
    """Compute comprehensive representation metrics."""
    results = {}
    
    # W+ metrics
    w_flat = w18.squeeze(0)  # [18, 512]
    w_centered = w_flat - w_flat.mean(dim=0)
    w_cov = w_centered.T @ w_centered / 17
    w_eigs = torch.linalg.eigvalsh(w_cov)
    w_eigs = w_eigs[w_eigs > 1e-10]
    
    results['w_plus'] = {
        'effective_rank': float((w_eigs.sum() ** 2 / (w_eigs ** 2).sum()).item()),
        'entropy': float(0.5 * (18 * np.log(2 * np.pi * np.e) + torch.log(w_eigs).sum()).item()) if len(w_eigs) > 0 else 0,
        'variance_explained': w_eigs.flip(0).cumsum(0) / w_eigs.sum(),
        'singular_values': w_eigs.flip(0).tolist(),
        'condition_number': float((w_eigs.max() / w_eigs.min()).item()) if len(w_eigs) > 1 and w_eigs.min() > 0 else -1,
        'trace': float(torch.trace(w_cov).item()),
        'frobenius_norm': float(torch.norm(w_cov, p='fro').item()),
    }
    
    # Sub34 metrics - reshape to [34, 18*512]
    sub34_flat = sub34.squeeze(0).permute(1, 0, 2).reshape(34, -1)  # [34, 9216]
    sub34_centered = sub34_flat - sub34_flat.mean(dim=1, keepdim=True)
    sub34_cov = sub34_centered @ sub34_centered.T / (sub34_flat.shape[1] - 1)  # [34, 34]
    sub34_eigs = torch.linalg.eigvalsh(sub34_cov)
    sub34_eigs = sub34_eigs[sub34_eigs > 1e-10]
    
    results['sub34'] = {
        'effective_rank': float((sub34_eigs.sum() ** 2 / (sub34_eigs ** 2).sum()).item()),
        'entropy': float(0.5 * (34 * np.log(2 * np.pi * np.e) + torch.log(sub34_eigs).sum()).item()) if len(sub34_eigs) > 0 else 0,
        'variance_explained': sub34_eigs.flip(0).cumsum(0) / sub34_eigs.sum(),
        'singular_values': sub34_eigs.flip(0).tolist(),
        'condition_number': float((sub34_eigs.max() / sub34_eigs.min()).item()) if len(sub34_eigs) > 1 and sub34_eigs.min() > 0 else -1,
        'trace': float(torch.trace(sub34_cov).item()),
        'frobenius_norm': float(torch.norm(sub34_cov, p='fro').item()),
    }
    
    # Z metrics (sampled latents)
    z_flat = z.squeeze(0).permute(1, 0, 2).reshape(34, -1)
    z_centered = z_flat - z_flat.mean(dim=1, keepdim=True)
    z_cov = z_centered @ z_centered.T / (z_flat.shape[1] - 1)
    z_eigs = torch.linalg.eigvalsh(z_cov)
    z_eigs = z_eigs[z_eigs > 1e-10]
    
    results['z'] = {
        'effective_rank': float((z_eigs.sum() ** 2 / (z_eigs ** 2).sum()).item()),
        'entropy': float(0.5 * (34 * np.log(2 * np.pi * np.e) + torch.log(z_eigs).sum()).item()) if len(z_eigs) > 0 else 0,
        'trace': float(torch.trace(z_cov).item()),
        'frobenius_norm': float(torch.norm(z_cov, p='fro').item()),
    }
    
    # Per-region metrics
    region_metrics = []
    for i in range(34):
        region_z = z.squeeze(0)[:, i, :]  # [18, 512]
        region_cov = (region_z - region_z.mean(dim=0)).T @ (region_z - region_z.mean(dim=0)) / 17
        region_eigs = torch.linalg.eigvalsh(region_cov)
        region_eigs = region_eigs[region_eigs > 1e-10]
        
        region_metrics.append({
            'region_idx': i,
            'effective_rank': float((region_eigs.sum() ** 2 / (region_eigs ** 2).sum()).item()) if len(region_eigs) > 0 else 0,
            'variance': float(torch.trace(region_cov).item()),
            'entropy': float(0.5 * (18 * np.log(2 * np.pi * np.e) + torch.log(region_eigs).sum()).item()) if len(region_eigs) > 0 else 0,
        })
    
    results['per_region'] = region_metrics
    
    # Null space analysis
    # Check if Sub34 has dimensions that don't affect W+
    # Compute Sub2W Jacobian approximation
    results['null_space'] = {
        'sub34_dim': 34 * 18 * 512,
        'w_plus_dim': 18 * 512,
        'expansion_ratio': (34 * 18 * 512) / (18 * 512),
    }
    
    # Mutual information between regions (approximate via correlation)
    region_corr = np.zeros((34, 34))
    for i in range(34):
        for j in range(34):
            if i != j:
                zi = z.squeeze(0)[:, i, :].flatten()
                zj = z.squeeze(0)[:, j, :].flatten()
                corr = torch.corrcoef(torch.stack([zi, zj]))[0, 1]
                region_corr[i, j] = float(corr.item())
    
    results['region_correlation_matrix'] = region_corr.tolist()
    results['mean_abs_correlation'] = float(np.mean(np.abs(region_corr[np.eye(34, dtype=bool) == False])))
    
    return results

def run_representation_analysis():
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
    
    all_results = []
    
    for f_file, m_file, gender, race_f, race_m, pair_name in TEST_PAIRS:
        print(f"\nProcessing {pair_name}...")
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        w18_F, aligned_F = load_and_encode(f_path, encoder, mean_latent, DEVICE)
        w18_M, aligned_M = load_and_encode(m_path, encoder, mean_latent, DEVICE)
        
        # Get regional latents
        with torch.no_grad():
            mu_F, var_F, z_F = w2sub34(w18_F)
            mu_M, var_M, z_M = w2sub34(w18_M)
            sub34_F = mu_F + var_F * z_F  # Reparameterized
            sub34_M = mu_M + var_M * z_M
        
        # Compute metrics for father
        metrics_F = compute_representation_metrics(w18_F, mu_F, var_F, z_F, sub34_F)
        metrics_M = compute_representation_metrics(w18_M, mu_M, var_M, z_M, sub34_M)
        
        result = {
            'pair': pair_name,
            'timestamp': datetime.now().isoformat(),
            'father': metrics_F,
            'mother': metrics_M,
        }
        all_results.append(result)
        
        print(f"  Father W+ eff_rank: {metrics_F['w_plus']['effective_rank']:.2f}, Sub34 eff_rank: {metrics_F['sub34']['effective_rank']:.2f}")
        print(f"  Mother W+ eff_rank: {metrics_M['w_plus']['effective_rank']:.2f}, Sub34 eff_rank: {metrics_M['sub34']['effective_rank']:.2f}")
        print(f"  Father region mean |corr|: {metrics_F['mean_abs_correlation']:.4f}")
        print(f"  Mother region mean |corr|: {metrics_M['mean_abs_correlation']:.4f}")
    
    # Aggregate
    print("\n" + "="*60)
    print("AGGREGATE REPRESENTATION ANALYSIS")
    print("="*60)
    
    for key in ['effective_rank', 'entropy', 'trace', 'condition_number']:
        w_vals = [r['father']['w_plus'][key] for r in all_results] + [r['mother']['w_plus'][key] for r in all_results]
        sub_vals = [r['father']['sub34'][key] for r in all_results] + [r['mother']['sub34'][key] for r in all_results]
        print(f"  {key}: W+={np.mean(w_vals):.4f}±{np.std(w_vals):.4f}, Sub34={np.mean(sub_vals):.4f}±{np.std(sub_vals):.4f}")
    
    # Per-region correlation
    corr_vals = [r['father']['mean_abs_correlation'] for r in all_results] + [r['mother']['mean_abs_correlation'] for r in all_results]
    print(f"  Mean |region correlation|: {np.mean(corr_vals):.4f}±{np.std(corr_vals):.4f}")
    
    # Variance explained comparison
    print("\n  Variance explained (95%):")
    for r in all_results:
        w_var = r['father']['w_plus']['variance_explained']
        if isinstance(w_var, list):
            w_95 = np.argmax(np.array(w_var) >= 0.95) + 1
        else:
            w_95 = np.argmax(w_var.cpu().numpy() >= 0.95) + 1
        sub_var = r['father']['sub34']['variance_explained']
        if isinstance(sub_var, list):
            sub_95 = np.argmax(np.array(sub_var) >= 0.95) + 1
        else:
            sub_95 = np.argmax(sub_var.cpu().numpy() >= 0.95) + 1
        print(f"    {r['pair']}: W+={w_95}, Sub34={sub_95}")
    
    # Save
    with open(OUTPUT_DIR / 'representation_analysis.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'all_results': all_results,
        }, f, indent=2)
    
    # CSV summary
    import csv
    with open(OUTPUT_DIR / 'rank_analysis.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pair', 'role', 'w_plus_eff_rank', 'sub34_eff_rank', 'w_plus_entropy', 'sub34_entropy', 'region_mean_corr'])
        for r in all_results:
            writer.writerow([r['pair'], 'father', r['father']['w_plus']['effective_rank'], r['father']['sub34']['effective_rank'], r['father']['w_plus']['entropy'], r['father']['sub34']['entropy'], r['father']['mean_abs_correlation']])
            writer.writerow([r['pair'], 'mother', r['mother']['w_plus']['effective_rank'], r['mother']['sub34']['effective_rank'], r['mother']['w_plus']['entropy'], r['mother']['sub34']['entropy'], r['mother']['mean_abs_correlation']])
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    return all_results

if __name__ == '__main__':
    run_representation_analysis()