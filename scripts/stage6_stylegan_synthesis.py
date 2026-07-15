"""
Stage 6: StyleGAN Synthesis Analysis
Do identical W+ always produce proportional geometry, or does generator amplify structural differences?
Measure layer influence on geometry.
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
    return {'mean_norm': mean_norm, 'std_norm': std_norm, 'mean_vector': mean_vec.cpu().numpy().tolist(), 'covariance_trace': cov_trace, 'effective_rank': eff_rank, 'entropy': entropy}

def layer_influence_analysis(generator, w18_base, w18_perturbed, geom_base):
    """Measure how each layer's difference affects geometry."""
    layers = []
    for k in range(18):
        # Create hybrid: base with only layer k from perturbed
        w18_hybrid = w18_base.clone()
        w18_hybrid[:, k, :] = w18_perturbed[:, k, :]
        
        with torch.no_grad():
            img, _ = generator([w18_hybrid], return_latents=True, input_is_latent=True)
        geom = compute_geometry(tensor2rgb(img))
        
        # Compute layer displacement
        l2_disp = float(torch.norm(w18_perturbed[:, k, :] - w18_base[:, k, :], p=2).item())
        cos_disp = float(F.cosine_similarity(w18_perturbed[:, k, :].flatten(), w18_base[:, k, :].flatten(), dim=0).item())
        
        # Geometry correlation
        geom_corr = {k: geom[k] - geom_base[k] for k in geom_base}
        
        layers.append({
            'layer_idx': k,
            'l2_displacement': l2_disp,
            'cosine_change': cos_disp,
            'geometry_delta': geom_corr,
        })
    return layers

def run_stage6():
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
        
        # Generate child W+
        w18_child = fuse_latent(
            w2sub34, sub2w, w18_F=w18_F, w18_M=w18_M,
            random_fakes=random_fakes, fixed_gamma=0.05, fixed_eta=0.4, arcs_lambda=0.0,
            child_gender='male', geometry_weight=0.7, texture_weight=0.5
        )
        
        # Generate baseline image
        with torch.no_grad():
            img_base, _ = generator([w18_child], return_latents=True, input_is_latent=True)
        geom_base = compute_geometry(tensor2rgb(img_base))
        
        # Test identical W+ multiple times (should be deterministic)
        with torch.no_grad():
            img_repeat1, _ = generator([w18_child], return_latents=True, input_is_latent=True)
            img_repeat2, _ = generator([w18_child], return_latents=True, input_is_latent=True)
        
        geom_r1 = compute_geometry(tensor2rgb(img_repeat1))
        geom_r2 = compute_geometry(tensor2rgb(img_repeat2))
        
        # Determinism check
        geom_match = all(abs(geom_r1[k] - geom_r2[k]) < 1e-5 for k in geom_r1 if geom_r1[k] > 0)
        
        # Layer influence analysis
        layer_influence = layer_influence_analysis(generator, w18_child, w18_F, geom_base)
        
        # Also test: perturb early vs late layers
        # Early layers (0-7): coarse structure
        # Mid layers (8-11): geometry
        # Late layers (12-17): texture
        
        w18_early_pert = w18_child.clone()
        w18_early_pert[:, :8, :] = w18_F[:, :8, :]  # Replace early layers with father
        with torch.no_grad():
            img_early, _ = generator([w18_early_pert], return_latents=True, input_is_latent=True)
        geom_early = compute_geometry(tensor2rgb(img_early))
        
        w18_mid_pert = w18_child.clone()
        w18_mid_pert[:, 8:12, :] = w18_F[:, 8:12, :]  # Replace geometry layers
        with torch.no_grad():
            img_mid, _ = generator([w18_mid_pert], return_latents=True, input_is_latent=True)
        geom_mid = compute_geometry(tensor2rgb(img_mid))
        
        w18_late_pert = w18_child.clone()
        w18_late_pert[:, 12:, :] = w18_F[:, 12:, :]  # Replace texture layers
        with torch.no_grad():
            img_late, _ = generator([w18_late_pert], return_latents=True, input_is_latent=True)
        geom_late = compute_geometry(tensor2rgb(img_late))
        
        # Perturbation magnitude analysis
        # Add small noise to W+ and measure geometry sensitivity
        noise_scales = [0.0, 0.1, 0.5, 1.0, 2.0]
        noise_sensitivity = []
        for scale in noise_scales:
            if scale == 0:
                w18_noisy = w18_child
            else:
                noise = torch.randn_like(w18_child) * scale
                w18_noisy = w18_child + noise
            with torch.no_grad():
                img_n, _ = generator([w18_noisy], return_latents=True, input_is_latent=True)
            geom_n = compute_geometry(tensor2rgb(img_n))
            noise_sensitivity.append({
                'noise_scale': scale,
                'wh_ratio': geom_n['wh_ratio'],
                'jaw_width': geom_n['jaw_width'],
                'cheek_width': geom_n['cheek_width'],
            })
        
        result = {
            'stage_name': 'StyleGAN_synthesis',
            'input_representation': 'W+ latent (18x512)',
            'output_representation': 'RGB image (1024x1024)',
            'pair': pair_name,
            'geometry_metrics': {
                'base': geom_base,
                'repeat1': geom_r1,
                'repeat2': geom_r2,
                'deterministic': geom_match,
                'early_layers_from_father': geom_early,
                'mid_layers_from_father': geom_mid,
                'late_layers_from_father': geom_late,
            },
            'identity_metrics': {
                'arcface_similarity_original': -1,
                'arcface_similarity_m_original': -1,
                'arcface_similarity_child': -1,
            },
            'latent_statistics': latent_stats(w18_child),
            'pca_analysis': None,
            'distortion_metrics': {
                'l2_distance_input_output': 0.0,
                'cosine_similarity_input_output': 1.0,
                'mahalanobis_distance': float(torch.norm(w18_child - mean_latent, p=2).item()),
                'latent_norm_drift': 0.0,
                'variance_reduction_ratio': -1,
                'covariance_frobenius_change': -1,
            },
            'layer_wise_analysis': {
                'layer_displacements': layer_influence,
                'early_vs_mid_vs_late': {
                    'early_layers_delta': {k: geom_early[k] - geom_base[k] for k in geom_base},
                    'mid_layers_delta': {k: geom_mid[k] - geom_base[k] for k in geom_base},
                    'late_layers_delta': {k: geom_late[k] - geom_base[k] for k in geom_base},
                },
                'noise_sensitivity': noise_sensitivity,
            },
            'region_wise_analysis': None,
            'visualizations_generated': [],
            'mathematical_findings': {
                'equation_analyzed': 'StyleGAN2 synthesis: W+ -> generator layers 0-17 -> RGB',
                'bias_detected': False,
                'bias_description': 'Test if identical W+ produces identical geometry (determinism) and which layers control geometry',
                'proof_summary': 'Test determinism, layer-wise replacement, and noise sensitivity across 5 pairs',
            },
            'hypothesis_result': 'INCONCLUSIVE',
            'contribution_score': 0.0,
            'evidence_summary': '',
            'code_references': [
                'StyleGene/models/stylegan2/model.py:Generator.forward()',
                'StyleGene/models/stylegene/api.py:generate_child()',
            ],
        }
        results.append(result)
        
        print(f"  Deterministic: {geom_match}")
        print(f"  wh_ratio: base={geom_base['wh_ratio']:.4f}, early={geom_early['wh_ratio']:.4f}, mid={geom_mid['wh_ratio']:.4f}, late={geom_late['wh_ratio']:.4f}")
        print(f"  Noise sensitivity: wh_ratio @ scale 2.0 = {noise_sensitivity[-1]['wh_ratio']:.4f}")
    
    # Aggregate
    print("\n=== AGGREGATE STATISTICS ===")
    print("Determinism check:", all(r['geometry_metrics']['deterministic'] for r in results))
    
    # Layer influence on wh_ratio
    for k in range(18):
        infls = [r['layer_wise_analysis']['layer_displacements'][k]['geometry_delta']['wh_ratio'] for r in results if 'wh_ratio' in r['layer_wise_analysis']['layer_displacements'][k]['geometry_delta']]
        if infls:
            print(f"  Layer {k}: wh_ratio influence={np.mean(infls):.4f}±{np.std(infls):.4f}")
    
    # Early/mid/late layer effects
    for label in ['early_layers_delta', 'mid_layers_delta', 'late_layers_delta']:
        deltas = [r['layer_wise_analysis']['early_vs_mid_vs_late'][label]['wh_ratio'] for r in results]
        if deltas:
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            d = np.mean(deltas) / np.std(deltas) if np.std(deltas) > 0 else 0
            print(f"  {label}: wh_ratio delta={np.mean(deltas):.4f}±{np.std(deltas):.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")
    
    # Noise sensitivity
    for scale in [0.1, 0.5, 1.0, 2.0]:
        whs = [r['layer_wise_analysis']['noise_sensitivity'][int(scale*10)]['wh_ratio'] for r in results if len(r['layer_wise_analysis']['noise_sensitivity']) > int(scale*10)]
        if whs:
            print(f"  Noise scale {scale}: wh_ratio={np.mean(whs):.4f}±{np.std(whs):.4f}")
    
    # Save
    output_dir = Path('C:/Users/mdiza/coding/KinshipForge-iz/kinshipforge-pipeline-bottleneck/results')
    output_dir.mkdir(exist_ok=True)
    
    for r in results:
        filename = f"Stage_6_StyleGAN_Synthesis_{r['pair']}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(r, f, indent=2)
    
    with open(output_dir / "Stage_6_StyleGAN_Synthesis_AGGREGATE.json", 'w') as f:
        json.dump({
            'stage_name': 'StyleGAN_synthesis',
            'summary': {'n_samples': len(results)},
            'all_results': results
        }, f, indent=2)
    
    print("\nStage 6 complete. Results saved.")
    return results

if __name__ == '__main__':
    import torch.nn.functional as F
    run_stage6()