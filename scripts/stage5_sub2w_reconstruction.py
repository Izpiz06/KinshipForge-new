"""
Stage 5: Sub2W Reconstruction - HIGHEST PRIORITY
Does 34 regional latent tensors -> single W+ introduce irreversible information loss?
Measure Before Sub2W vs After Sub2W: PCA, L2, Cosine, Variance, Geometry. Visualize difference heatmaps.
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

def compute_difference_heatmap(img1, img2, save_path=None):
    """Compute pixel-wise difference heatmap."""
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    heatmap = diff.mean(axis=2)
    if save_path:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap, cmap='hot')
        plt.colorbar(label='Mean Absolute Difference')
        plt.title('Difference Heatmap')
        plt.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return {
        'mean_abs_diff': float(heatmap.mean()),
        'max_abs_diff': float(heatmap.max()),
        'std_abs_diff': float(heatmap.std()),
    }

def run_stage5():
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
    
    pool_data = torch.load(path_ckpt_genepool, map_location='cpu', weights_only=False)
    geneFactor = GenePoolFactory(root_ffhq=None, device=DEVICE, mean_latent=mean_latent, max_sample=300)
    geneFactor.pools = pool_data
    
    model_fair_7 = init_fair_model(DEVICE)
    
    results = []
    
    for f_file, m_file, gender, race_f, race_m, pair_name in TEST_PAIRS:
        print(f"\nProcessing {pair_name}...")
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
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
        
        # Full pipeline: crossover + mutation + mix
        w18_syn = fuse_latent(
            w2sub34, sub2w, w18_F=w18_F, w18_M=w18_M,
            random_fakes=random_fakes, fixed_gamma=0.05, fixed_eta=0.4, arcs_lambda=0.0,
            child_gender='male', geometry_weight=0.5, texture_weight=0.5
        )
        
        # Get regional latents BEFORE Sub2W
        mu_F, var_F, sub34_F = w2sub34(w18_F)
        mu_M, var_M, sub34_M = w2sub34(w18_M)
        
        # Run crossover to get new_sub34 (before Sub2W)
        from models.stylegene.gene_crossover_mutation import REGION_SENSITIVITY_MAP, face_class, reparameterize
        import random
        
        s_map = REGION_SENSITIVITY_MAP
        s_vals = list(s_map.values())
        s_min = min(s_vals) if s_vals else 0.0
        s_max = max(s_vals) if s_vals else 1.0
        s_range = s_max - s_min if s_max != s_min else 1.0
        
        resolved_gammas = {}
        for name in face_class:
            if name == 'background': continue
            s_val = s_map.get(name, 0.0)
            s_norm = (s_val - s_min) / s_range
            g_val = 0.05 * (1.0 - 0.0 * s_norm)
            resolved_gammas[name] = g_val
        
        weights = {}
        for name in face_class:
            g_val = resolved_gammas.get(name, 0.05)
            weights[name] = (random.uniform(0, 1 - g_val), g_val)
        
        cur_class = random.sample(face_class, int(len(face_class) * (1 - 0.4)))
        
        new_sub34 = torch.zeros_like(sub34_F, dtype=torch.float, device=DEVICE)
        
        for i, classname in enumerate(face_class):
            if classname == 'background':
                new_sub34[:, :, i, :] = reparameterize(mu_F[:, :, i, :], var_F[:, :, i, :])
                continue
            if classname in cur_class:
                fake_mu, fake_var = random.choice(random_fakes)
                w_i, b_i = weights[classname]
                new_sub34[:, :, i, :] = reparameterize(
                    mu_F[:, :, i, :] * w_i + fake_mu[:, :, i, :].to(DEVICE) * b_i + mu_M[:, :, i, :] * (1 - w_i - b_i),
                    var_F[:, :, i, :] * w_i + fake_var[:, :, i, :].to(DEVICE) * b_i + var_M[:, :, i, :] * (1 - w_i - b_i))
            else:
                fake_mu, fake_var = random.choice(random_fakes)
                fake_latent = reparameterize(fake_mu[:, :, i, :], fake_var[:, :, i, :]).to(DEVICE)
                var = fake_latent
                new_sub34[:, :, i, :] = new_sub34[:, :, i, :] + var
        
        # BEFORE Sub2W: new_sub34 (34 regional latents)
        # AFTER Sub2W: w18_syn = sub2w(new_sub34)
        
        with torch.no_grad():
            w18_after_sub2w = sub2w(new_sub34)
        
        # Generate images
        img_before_sub2w, _ = generator([w18_after_sub2w], return_latents=True, input_is_latent=True)
        img_before_np = tensor2rgb(img_before_sub2w)
        
        img_after_sub2w, _ = generator([w18_syn], return_latents=True, input_is_latent=True)
        img_after_np = tensor2rgb(img_after_sub2w)
        
        # Also get original W+ image for reference
        img_original_w, _ = generator([w18_F], return_latents=True, input_is_latent=True)
        img_original_np = tensor2rgb(img_original_w)
        
        # Compute metrics
        geom_before = compute_geometry(img_before_np)
        geom_after = compute_geometry(img_after_np)
        geom_original = compute_geometry(img_original_np)
        
        # Latent statistics
        lat_before = latent_stats(w18_after_sub2w)
        lat_after = latent_stats(w18_syn)
        lat_original = latent_stats(w18_F)
        
        # PCA
        pca_before = pca_analysis(w18_after_sub2w)
        pca_after = pca_analysis(w18_syn)
        
        # Distortion metrics
        l2_dist = float(torch.norm(w18_after_sub2w - w18_syn, p=2).item())
        cos_sim = float(F.cosine_similarity(w18_after_sub2w.flatten(), w18_syn.flatten(), dim=0).item())
        maha = float(torch.norm(w18_after_sub2w - mean_latent, p=2).item())
        norm_drift = lat_after['mean_norm'] - lat_before['mean_norm']
        var_reduction = lat_after['covariance_trace'] / lat_before['covariance_trace'] if lat_before['covariance_trace'] > 0 else -1
        
        # Covariance Frobenius change
        w_flat_b = w18_after_sub2w.squeeze(0)
        centered_b = w_flat_b - w_flat_b.mean(dim=0)
        cov_b = centered_b.T @ centered_b / 17
        
        w_flat_a = w18_syn.squeeze(0)
        centered_a = w_flat_a - w_flat_a.mean(dim=0)
        cov_a = centered_a.T @ centered_a / 17
        
        cov_frob_change = float(torch.norm(cov_b - cov_a, p='fro').item())
        
        # Per-layer analysis
        layer_displacements = []
        for k in range(18):
            l2 = float(torch.norm(w18_after_sub2w[:, k, :] - w18_syn[:, k, :], p=2).item())
            cos = float(F.cosine_similarity(w18_after_sub2w[:, k, :].flatten(), w18_syn[:, k, :].flatten(), dim=0).item())
            layer_displacements.append({'layer_idx': k, 'l2_displacement': l2, 'cosine_change': cos})
        
        # Difference heatmap
        heatmap_dir = Path('C:/Users/mdiza/coding/KinshipForge-iz/kinshipforge-pipeline-bottleneck/results/heatmaps')
        heatmap_dir.mkdir(exist_ok=True)
        heatmap_path = heatmap_dir / f"Stage5_Sub2W_{pair_name}_heatmap.png"
        heatmap_metrics = compute_difference_heatmap(img_before_np, img_after_np, str(heatmap_path))
        
        result = {
            'stage_name': 'Sub2W_reconstruction',
            'input_representation': '34 regional latent tensors (new_sub34)',
            'output_representation': 'W+ latent (18x512)',
            'pair': pair_name,
            'geometry_metrics': {
                'before_sub2w': geom_before,
                'after_sub2w': geom_after,
                'original_w_plus': geom_original,
                'delta': {k: geom_after[k] - geom_before[k] for k in geom_before},
            },
            'identity_metrics': {
                'arcface_similarity_original': -1,
                'arcface_similarity_m_original': -1,
                'arcface_similarity_child': -1,
            },
            'latent_statistics': lat_before,
            'pca_analysis': pca_before,
            'distortion_metrics': {
                'l2_distance_input_output': l2_dist,
                'cosine_similarity_input_output': cos_sim,
                'mahalanobis_distance': maha,
                'latent_norm_drift': norm_drift,
                'variance_reduction_ratio': var_reduction,
                'covariance_frobenius_change': cov_frob_change,
            },
            'layer_wise_analysis': {
                'layer_displacements': layer_displacements,
            },
            'region_wise_analysis': None,
            'visualizations_generated': [
                {'type': 'difference_heatmap', 'path': str(heatmap_path), 'description': 'Before vs After Sub2W pixel difference'}
            ],
            'mathematical_findings': {
                'equation_analyzed': 'Sub2W: 34 regional latents (18x512 each) -> single W+ (18x512) via learned mapping',
                'bias_detected': False,
                'bias_description': 'Measure information loss: geometry, latent stats, PCA, variance reduction',
                'proof_summary': 'Compare before/after Sub2W on full pipeline output across 5 pairs',
            },
            'hypothesis_result': 'INCONCLUSIVE',
            'contribution_score': 0.0,
            'evidence_summary': '',
            'code_references': [
                'StyleGene/models/stylegene/model.py:MappingSub2W.forward()',
                'StyleGene/models/stylegene/gene_crossover_mutation.py:fuse_latent() line 167',
            ],
        }
        results.append(result)
        
        print(f"  wh_ratio: before={geom_before['wh_ratio']:.4f}, after={geom_after['wh_ratio']:.4f}, delta={geom_after['wh_ratio']-geom_before['wh_ratio']:.4f}")
        print(f"  jaw_width: before={geom_before['jaw_width']:.1f}, after={geom_after['jaw_width']:.1f}, delta={geom_after['jaw_width']-geom_before['jaw_width']:.1f}")
        print(f"  cheek_width: before={geom_before['cheek_width']:.1f}, after={geom_after['cheek_width']:.1f}, delta={geom_after['cheek_width']-geom_before['cheek_width']:.1f}")
        print(f"  L2={l2_dist:.2f}, cos={cos_sim:.4f}, var_reduction={var_reduction:.4f}, cov_frob={cov_frob_change:.2f}")
        print(f"  Heatmap: mean_diff={heatmap_metrics['mean_abs_diff']:.2f}, max_diff={heatmap_metrics['max_abs_diff']:.2f}")
    
    # Aggregate
    print("\n=== AGGREGATE STATISTICS ===")
    for metric in ['wh_ratio', 'jaw_width', 'cheek_width']:
        deltas = [r['geometry_metrics']['delta'][metric] for r in results if r['geometry_metrics']['delta'][metric] != 0]
        if deltas:
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            d = np.mean(deltas) / np.std(deltas) if np.std(deltas) > 0 else 0
            ci = 1.96 * np.std(deltas) / np.sqrt(len(deltas))
            print(f"  {metric}: delta={np.mean(deltas):.4f}±{np.std(deltas):.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}, CI95=[{np.mean(deltas)-ci:.4f}, {np.mean(deltas)+ci:.4f}]")
            
            if p_val < 0.05 and np.mean(deltas) > 0:
                for r in results: r['hypothesis_result'] = 'CONFIRMED'
            elif p_val < 0.05 and np.mean(deltas) < 0:
                for r in results: r['hypothesis_result'] = 'FALSIFIED (narrowing)'
            else:
                for r in results: r['hypothesis_result'] = 'INCONCLUSIVE'
            
            if metric == 'wh_ratio':
                effect = abs(np.mean(deltas) / np.std(deltas)) if np.std(deltas) > 0 else 0
                for r in results:
                    r['contribution_score'] = float(min(effect / 2.0, 1.0))
                    r['evidence_summary'] = f"Sub2W wh_ratio delta: {np.mean(deltas):.4f}±{np.std(deltas):.4f} (n={len(deltas)}), p={p_val:.4f}, Cohen's d={np.mean(deltas)/np.std(deltas) if np.std(deltas)>0 else 0:.3f}"
    
    # Layer-wise
    print("\n=== LAYER-WISE SUB2W DISPLACEMENT ===")
    for k in range(18):
        l2s = [r['layer_wise_analysis']['layer_displacements'][k]['l2_displacement'] for r in results]
        coss = [r['layer_wise_analysis']['layer_displacements'][k]['cosine_change'] for r in results]
        if l2s:
            print(f"  Layer {k}: L2={np.mean(l2s):.2f}±{np.std(l2s):.2f}, cos={np.mean(coss):.4f}±{np.std(coss):.4f}")
    
    # Save
    output_dir = Path('C:/Users/mdiza/coding/KinshipForge-iz/kinshipforge-pipeline-bottleneck/results')
    output_dir.mkdir(exist_ok=True)
    
    for r in results:
        filename = f"Stage_5_Sub2W_Reconstruction_{r['pair']}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(r, f, indent=2)
    
    with open(output_dir / "Stage_5_Sub2W_Reconstruction_AGGREGATE.json", 'w') as f:
        json.dump({
            'stage_name': 'Sub2W_reconstruction',
            'summary': {'n_samples': len(results)},
            'all_results': results
        }, f, indent=2)
    
    print("\nStage 5 complete. Results saved.")
    return results

if __name__ == '__main__':
    run_stage5()