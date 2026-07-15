"""
Task 2: Mathematical Invertibility Test for W2Sub/Sub2W
Measures: L2 reconstruction error, Cosine similarity, Variance preservation, PCA, Covariance, Rank, Singular value spectrum, Mutual information
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
from datetime import datetime
import tempfile

sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegene.api import init_model, tensor2rgb, load_img
from models.stylegene.model import MappingW2Sub, MappingSub2W
from preprocess.align_images import align_face
from configs import path_ckpt_e4e, path_ckpt_stylegan2, path_ckpt_stylegene
from scripts.legacy.geometry_utils import GeometryEstimator

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
geom_estimator = GeometryEstimator()

PHOTOS = 'C:/Users/mdiza/coding/KinshipForge-iz/archive'
TEST_PAIRS = [
    ("father_p1.jpg", "mother_p1.jpg", "P1_Shahrukh_Gauri"),
    ("father_p2.jpg", "mother_p2.jpeg", "P2_Jackie_Joan"),
    ("father_p3.jpg", "mother_p3.jpeg", "P3_Obama_Michelle"),
    ("father_p4.jpg", "mother_p4.jpg", "P4_TomHanks_Rita"),
    ("father_p5.jpg", "mother_p5.jpg", "P5_Ben_Laura"),
]

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/w2sub_sub2w_invertibility')
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'intermediate_latents').mkdir(exist_ok=True)
(OUTPUT_DIR / 'intermediate_images').mkdir(exist_ok=True)
(OUTPUT_DIR / 'plots').mkdir(exist_ok=True)
(OUTPUT_DIR / 'covariance').mkdir(exist_ok=True)
(OUTPUT_DIR / 'pca').mkdir(exist_ok=True)
(OUTPUT_DIR / 'heatmaps').mkdir(exist_ok=True)

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

def latent_statistics(w18):
    """Comprehensive latent statistics."""
    with torch.no_grad():
        w_flat = w18.squeeze(0)  # [18, 512]
        norms = torch.norm(w_flat, p=2, dim=-1)
        mean_norm = float(norms.mean().item())
        std_norm = float(norms.std().item())
        
        mean_vec = w_flat.mean(dim=0)
        centered = w_flat - mean_vec
        cov = centered.T @ centered / (w_flat.shape[0] - 1)
        cov_trace = float(torch.trace(cov).item())
        
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        if len(eigvals) > 0:
            eff_rank = float((eigvals.sum() ** 2 / (eigvals ** 2).sum()).item())
            entropy = float(0.5 * (18 * np.log(2 * np.pi * np.e) + torch.log(eigvals).sum()).item())
            sv_spectrum = eigvals.flip(0).cpu().numpy().tolist()
        else:
            eff_rank = 0.0
            entropy = 0.0
            sv_spectrum = []
        
        # Frobenius norm of covariance
        cov_frob = float(torch.norm(cov, p='fro').item())
        
        # Condition number
        cond_num = float((eigvals.max() / eigvals.min()).item()) if len(eigvals) > 1 and eigvals.min() > 0 else -1
        
    return {
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'covariance_trace': cov_trace,
        'covariance_frobenius': cov_frob,
        'effective_rank': eff_rank,
        'entropy': entropy,
        'singular_value_spectrum': sv_spectrum,
        'condition_number': cond_num,
        'mean_vector': mean_vec.cpu().numpy().tolist(),
    }

def pca_analysis(w18, n_components=18):
    """PCA on W+ latent."""
    from sklearn.decomposition import PCA
    w_flat = w18.squeeze(0).cpu().numpy()  # [18, 512]
    pca = PCA(n_components=min(n_components, 18))
    pca.fit(w_flat)
    explained_var = pca.explained_variance_ratio_.tolist()
    cumsum_var = np.cumsum(explained_var).tolist()
    n_95 = int(np.argmax(np.array(cumsum_var) >= 0.95)) + 1
    n_99 = int(np.argmax(np.array(cumsum_var) >= 0.99)) + 1
    return {
        'explained_variance_ratio': explained_var,
        'cumulative_variance': cumsum_var,
        'n_components_95': n_95,
        'n_components_99': n_99,
        'components': pca.components_.tolist(),
        'singular_values': pca.singular_values_.tolist(),
    }

def reconstruction_metrics(w18_orig, w18_recon):
    """L2, cosine, variance preservation, etc."""
    l2 = float(torch.norm(w18_orig - w18_recon, p=2).item())
    cos = float(F.cosine_similarity(w18_orig.flatten(), w18_recon.flatten(), dim=0).item())
    
    stats_orig = latent_statistics(w18_orig)
    stats_recon = latent_statistics(w18_recon)
    
    var_preservation = stats_recon['covariance_trace'] / stats_orig['covariance_trace'] if stats_orig['covariance_trace'] > 0 else -1
    rank_preservation = stats_recon['effective_rank'] / stats_orig['effective_rank'] if stats_orig['effective_rank'] > 0 else -1
    entropy_diff = stats_recon['entropy'] - stats_orig['entropy']
    
    # Per-layer metrics
    layer_l2 = []
    layer_cos = []
    for k in range(18):
        l2_k = float(torch.norm(w18_orig[:, k, :] - w18_recon[:, k, :], p=2).item())
        cos_k = float(F.cosine_similarity(w18_orig[:, k, :].flatten(), w18_recon[:, k, :].flatten(), dim=0).item())
        layer_l2.append(l2_k)
        layer_cos.append(cos_k)
    
    return {
        'l2_distance': l2,
        'cosine_similarity': cos,
        'variance_preservation_ratio': var_preservation,
        'rank_preservation_ratio': rank_preservation,
        'entropy_difference': entropy_diff,
        'layer_l2': layer_l2,
        'layer_cosine': layer_cos,
    }

def mahalanobis_distance(w18, mean_latent):
    """Mahalanobis distance from FFHQ mean."""
    diff = w18 - mean_latent
    return float(torch.norm(diff, p=2).item())

def drift_metrics(w18_orig, w18_recon, mean_latent):
    """Measure drift toward FFHQ mean."""
    maha_orig = mahalanobis_distance(w18_orig, mean_latent)
    maha_recon = mahalanobis_distance(w18_recon, mean_latent)
    norm_orig = float(torch.norm(w18_orig, p=2).item())
    norm_recon = float(torch.norm(w18_recon, p=2).item())
    
    return {
        'mahalanobis_orig': maha_orig,
        'mahalanobis_recon': maha_recon,
        'mahalanobis_drift': maha_recon - maha_orig,
        'latent_norm_orig': norm_orig,
        'latent_norm_recon': norm_recon,
        'norm_drift': norm_recon - norm_orig,
    }

def save_latent(w18, path):
    torch.save(w18.cpu(), path)

def save_image(img_tensor, path):
    img_np = tensor2rgb(img_tensor)
    Image.fromarray(img_np).save(path)

def difference_heatmap(img1, img2, save_path):
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    heatmap = diff.mean(axis=2)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar(label='Mean Abs Difference')
    plt.title('Difference Heatmap')
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return {'mean_abs_diff': float(heatmap.mean()), 'max_abs_diff': float(heatmap.max())}

def run_invertibility_experiment():
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
    
    for f_file, m_file, pair_name in TEST_PAIRS:
        print(f"\n{'='*60}")
        print(f"Processing {pair_name}...")
        print(f"{'='*60}")
        
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        for role, img_path in [("father", f_path), ("mother", m_path)]:
            # Load and encode
            w18_orig, aligned = load_and_encode(img_path, encoder, mean_latent, DEVICE)
            
            # Save original latent
            save_latent(w18_orig, OUTPUT_DIR / 'intermediate_latents' / f"{pair_name}_{role}_w18_orig.pt")
            
            # Generate original image
            with torch.no_grad():
                img_orig, _ = generator([w18_orig], return_latents=True, input_is_latent=True)
            img_orig_np = tensor2rgb(img_orig)
            save_image(img_orig, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_{role}_orig.png")
            geom_orig = compute_geometry(img_orig_np)
            
            # W2Sub forward pass
            with torch.no_grad():
                mu, var, z = w2sub34(w18_orig)
            
            save_latent(mu, OUTPUT_DIR / 'intermediate_latents' / f"{pair_name}_{role}_mu.pt")
            save_latent(var, OUTPUT_DIR / 'intermediate_latents' / f"{pair_name}_{role}_var.pt")
            save_latent(z, OUTPUT_DIR / 'intermediate_latents' / f"{pair_name}_{role}_z.pt")
            
            # Sub2W reconstruction
            with torch.no_grad():
                w18_recon = sub2w(z)
            
            save_latent(w18_recon, OUTPUT_DIR / 'intermediate_latents' / f"{pair_name}_{role}_w18_recon.pt")
            
            # Generate reconstructed image
            with torch.no_grad():
                img_recon, _ = generator([w18_recon], return_latents=True, input_is_latent=True)
            img_recon_np = tensor2rgb(img_recon)
            save_image(img_recon, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_{role}_recon.png")
            geom_recon = compute_geometry(img_recon_np)
            
            # Difference heatmap
            heatmap_path = OUTPUT_DIR / 'heatmaps' / f"{pair_name}_{role}_diff_heatmap.png"
            heatmap_metrics = difference_heatmap(img_orig_np, img_recon_np, heatmap_path)
            
            # Compute all metrics
            stats_orig = latent_statistics(w18_orig)
            stats_recon = latent_statistics(w18_recon)
            pca_orig = pca_analysis(w18_orig)
            pca_recon = pca_analysis(w18_recon)
            recon_metrics = reconstruction_metrics(w18_orig, w18_recon)
            drift = drift_metrics(w18_orig, w18_recon, mean_latent)
            
            # Covariance matrices
            w_flat_orig = w18_orig.squeeze(0)
            centered_orig = w_flat_orig - w_flat_orig.mean(dim=0)
            cov_orig = centered_orig.T @ centered_orig / 17
            
            w_flat_recon = w18_recon.squeeze(0)
            centered_recon = w_flat_recon - w_flat_recon.mean(dim=0)
            cov_recon = centered_recon.T @ centered_recon / 17
            
            cov_frob_diff = float(torch.norm(cov_orig - cov_recon, p='fro').item())
            
            # Eigenvalue analysis
            eig_orig = torch.linalg.eigvalsh(cov_orig)
            eig_recon = torch.linalg.eigvalsh(cov_recon)
            eig_orig = eig_orig[eig_orig > 1e-10]
            eig_recon = eig_recon[eig_recon > 1e-10]
            
            # Save covariance matrices
            torch.save(cov_orig.cpu(), OUTPUT_DIR / 'covariance' / f"{pair_name}_{role}_cov_orig.pt")
            torch.save(cov_recon.cpu(), OUTPUT_DIR / 'covariance' / f"{pair_name}_{role}_cov_recon.pt")
            
            result = {
                'pair': pair_name,
                'role': role,
                'timestamp': datetime.now().isoformat(),
                'geometry': {
                    'original': geom_orig,
                    'reconstructed': geom_recon,
                    'delta': {k: geom_recon[k] - geom_orig[k] for k in geom_orig},
                },
                'latent_statistics': {
                    'original': stats_orig,
                    'reconstructed': stats_recon,
                },
                'pca': {
                    'original': pca_orig,
                    'reconstructed': pca_recon,
                },
                'reconstruction_metrics': recon_metrics,
                'drift_metrics': drift,
                'covariance_frobenius_diff': cov_frob_diff,
                'eigenvalue_analysis': {
                    'original': eig_orig.tolist(),
                    'reconstructed': eig_recon.tolist(),
                    'eigenvalue_ratio': (eig_recon / eig_orig).tolist() if len(eig_orig) == len(eig_recon) else [],
                },
                'heatmap_metrics': heatmap_metrics,
            }
            all_results.append(result)
            
            # Print summary
            print(f"  {role}: L2={recon_metrics['l2_distance']:.4f}, Cos={recon_metrics['cosine_similarity']:.6f}")
            print(f"  wh_ratio: orig={geom_orig['wh_ratio']:.4f}, recon={geom_recon['wh_ratio']:.4f}, delta={geom_recon['wh_ratio']-geom_orig['wh_ratio']:.4f}")
            print(f"  jaw: orig={geom_orig['jaw_width']:.1f}, recon={geom_recon['jaw_width']:.1f}")
            print(f"  cheek: orig={geom_orig['cheek_width']:.1f}, recon={geom_recon['cheek_width']:.1f}")
            print(f"  var_preservation={recon_metrics['variance_preservation_ratio']:.4f}, rank_preservation={recon_metrics['rank_preservation_ratio']:.4f}")
            print(f"  mahalanobis_drift={drift['mahalanobis_drift']:.4f}, norm_drift={drift['norm_drift']:.4f}")
            print(f"  heatmap_mean_diff={heatmap_metrics['mean_abs_diff']:.2f}")
    
    # Aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS")
    print("="*60)
    
    metrics = ['l2_distance', 'cosine_similarity', 'variance_preservation_ratio', 'rank_preservation_ratio', 'entropy_difference']
    for metric in metrics:
        vals = [r['reconstruction_metrics'][metric] for r in all_results]
        print(f"  {metric}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, min={np.min(vals):.6f}, max={np.max(vals):.6f}")
    
    geom_metrics = ['wh_ratio', 'jaw_width', 'cheek_width']
    for metric in geom_metrics:
        deltas = [r['geometry']['delta'][metric] for r in all_results if r['geometry']['delta'][metric] != 0]
        if deltas:
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            d = np.mean(deltas) / np.std(deltas) if np.std(deltas) > 0 else 0
            ci = 1.96 * np.std(deltas) / np.sqrt(len(deltas))
            print(f"  {metric} delta: mean={np.mean(deltas):.4f}, std={np.std(deltas):.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}, CI95=[{np.mean(deltas)-ci:.4f}, {np.mean(deltas)+ci:.4f}]")
    
    drift_vals = [r['drift_metrics']['mahalanobis_drift'] for r in all_results]
    norm_drifts = [r['drift_metrics']['norm_drift'] for r in all_results]
    t_maha, p_maha = stats.ttest_1samp(drift_vals, 0)
    t_norm, p_norm = stats.ttest_1samp(norm_drifts, 0)
    print(f"  mahalanobis_drift: mean={np.mean(drift_vals):.4f}, t={t_maha:.3f}, p={p_maha:.4f}")
    print(f"  norm_drift: mean={np.mean(norm_drifts):.4f}, t={t_norm:.3f}, p={p_norm:.4f}")
    
    # Save aggregate results
    with open(OUTPUT_DIR / 'reconstruction_metrics.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(all_results),
            'aggregate': {
                'reconstruction_metrics': {m: {'mean': float(np.mean([r['reconstruction_metrics'][m] for r in all_results])), 'std': float(np.std([r['reconstruction_metrics'][m] for r in all_results]))} for m in metrics},
                'geometry_deltas': {m: {'mean': float(np.mean([r['geometry']['delta'][m] for r in all_results if r['geometry']['delta'][m] != 0])), 'std': float(np.std([r['geometry']['delta'][m] for r in all_results if r['geometry']['delta'][m] != 0]))} for m in geom_metrics},
                'mahalanobis_drift': {'mean': float(np.mean(drift_vals)), 'std': float(np.std(drift_vals)), 't_stat': float(t_maha), 'p_value': float(p_maha)},
                'norm_drift': {'mean': float(np.mean(norm_drifts)), 'std': float(np.std(norm_drifts)), 't_stat': float(t_norm), 'p_value': float(p_norm)},
            },
            'all_results': all_results
        }, f, indent=2)
    
    # Save CSV for easy analysis
    import csv
    with open(OUTPUT_DIR / 'latent_statistics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pair', 'role', 'l2', 'cosine', 'var_preservation', 'rank_preservation', 'entropy_diff', 'mahalanobis_drift', 'norm_drift', 'wh_ratio_delta', 'jaw_delta', 'cheek_delta'])
        for r in all_results:
            writer.writerow([
                r['pair'], r['role'],
                r['reconstruction_metrics']['l2_distance'],
                r['reconstruction_metrics']['cosine_similarity'],
                r['reconstruction_metrics']['variance_preservation_ratio'],
                r['reconstruction_metrics']['rank_preservation_ratio'],
                r['reconstruction_metrics']['entropy_difference'],
                r['drift_metrics']['mahalanobis_drift'],
                r['drift_metrics']['norm_drift'],
                r['geometry']['delta']['wh_ratio'],
                r['geometry']['delta']['jaw_width'],
                r['geometry']['delta']['cheek_width'],
            ])
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    return all_results

if __name__ == '__main__':
    run_invertibility_experiment()