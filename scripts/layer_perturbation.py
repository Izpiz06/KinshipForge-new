"""
Task 3: Layer-wise Perturbation Analysis
Inject perturbations into individual W+ layers and observe propagation through W2Sub -> Sub2W
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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/w2sub_sub2w_invertibility/layer_perturbation')
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'heatmaps').mkdir(exist_ok=True)

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

def run_layer_perturbation():
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
        print(f"\nProcessing {pair_name}...")
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        # Load and encode father
        w18_F, aligned_F = load_and_encode(f_path, encoder, mean_latent, DEVICE)
        
        # Get baseline geometry from original W+
        with torch.no_grad():
            img_baseline, _ = generator([w18_F], return_latents=True, input_is_latent=True)
        geom_baseline = compute_geometry(tensor2rgb(img_baseline))
        
        # Get gene pools for crossover
        race_F_det, _, _, _ = predict_race(model_fair_7, 
            torch.from_numpy(aligned_F.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)/255.0, DEVICE)
        
        pool_F = geneFactor(encoder, w2sub34(w18_F)[2], POOL_AGE, gender, race_F_det)
        if not pool_F:
            for age in ['0-2', '3-9', '10-19', '20-29']:
                if age != POOL_AGE:
                    pool_F += geneFactor(encoder, w2sub34(w18_F)[2], age, gender, race_F_det)
        
        # For controlled test, use father's own pool as gene pool (self-crossover)
        from models.stylegene.api import brdas_sampler
        random_fakes = brdas_sampler(pool_F, pool_F, 0.5, 0.5)
        
        # Generate baseline child (full pipeline)
        w18_child = fuse_latent(
            w2sub34, sub2w, w18_F=w18_F, w18_M=w18_F,
            random_fakes=random_fakes, fixed_gamma=0.05, fixed_eta=0.4, arcs_lambda=0.0,
            child_gender='male', geometry_weight=0.7, texture_weight=0.5
        )
        
        with torch.no_grad():
            img_child, _ = generator([w18_child], return_latents=True, input_is_latent=True)
        geom_child = compute_geometry(tensor2rgb(img_child))
        
        # PERTURBATION EXPERIMENTS
        perturbation_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        layer_results = []
        
        for layer_idx in range(18):
            layer_data = {'layer_idx': layer_idx, 'perturbations': []}
            
            for scale in perturbation_scales:
                # Create perturbation on specific layer
                w18_pert = w18_F.clone()
                noise = torch.randn_like(w18_pert[:, layer_idx, :]) * scale
                w18_pert[:, layer_idx, :] += noise
                
                # Pass through W2Sub -> Sub2W
                with torch.no_grad():
                    mu, var, z = w2sub34(w18_pert)
                    w18_recon = sub2w(z)
                
                # Measure latent reconstruction
                l2_recon = float(torch.norm(w18_pert - w18_recon, p=2).item())
                cos_recon = float(F.cosine_similarity(w18_pert.flatten(), w18_recon.flatten(), dim=0).item())
                
                # Per-layer displacement
                layer_disp = float(torch.norm(w18_pert[:, layer_idx, :] - w18_recon[:, layer_idx, :], p=2).item())
                layer_cos = float(F.cosine_similarity(
                    w18_pert[:, layer_idx, :].flatten(), 
                    w18_recon[:, layer_idx, :].flatten(), dim=0).item())
                
                # Generate image
                with torch.no_grad():
                    img_pert, _ = generator([w18_recon], return_latents=True, input_is_latent=True)
                geom_pert = compute_geometry(tensor2rgb(img_pert))
                
                # Geometry delta from baseline
                geom_delta = {k: geom_pert[k] - geom_baseline[k] for k in geom_baseline}
                
                layer_data['perturbations'].append({
                    'scale': scale,
                    'l2_reconstruction': l2_recon,
                    'cosine_reconstruction': cos_recon,
                    'layer_displacement': layer_disp,
                    'layer_cosine': layer_cos,
                    'geometry_delta': geom_delta,
                })
            
            layer_results.append(layer_data)
            
            # Print summary for this layer
            last_pert = layer_data['perturbations'][-1]  # scale=5.0
            print(f"  Layer {layer_idx:2d}: L2_recon={last_pert['l2_reconstruction']:.2f}, "
                  f"cos={last_pert['cosine_reconstruction']:.4f}, "
                  f"layer_disp={last_pert['layer_displacement']:.2f}, "
                  f"wh_delta={last_pert['geometry_delta']['wh_ratio']:.4f}, "
                  f"jaw_delta={last_pert['geometry_delta']['jaw_width']:.1f}")
        
        # Also test cross-layer coupling: perturb layer i, measure effect on layer j
        cross_layer = np.zeros((18, 18))
        for i in range(18):
            w18_pert = w18_F.clone()
            w18_pert[:, i, :] += torch.randn_like(w18_pert[:, i, :]) * 2.0
            with torch.no_grad():
                mu, var, z = w2sub34(w18_pert)
                w18_recon = sub2w(z)
            for j in range(18):
                disp = float(torch.norm(w18_pert[:, j, :] - w18_recon[:, j, :], p=2).item())
                cross_layer[i, j] = disp
        
        result = {
            'pair': pair_name,
            'timestamp': datetime.now().isoformat(),
            'baseline_geometry': geom_baseline,
            'child_geometry': geom_child,
            'layer_perturbations': layer_results,
            'cross_layer_coupling': cross_layer.tolist(),
        }
        all_results.append(result)
        
        # Save per-pair
        with open(OUTPUT_DIR / f"{pair_name}_layer_perturbation.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    # Aggregate cross-layer coupling
    print("\n" + "="*60)
    print("AGGREGATE CROSS-LAYER COUPLING")
    print("="*60)
    
    agg_coupling = np.zeros((18, 18))
    for r in all_results:
        agg_coupling += np.array(r['cross_layer_coupling'])
    agg_coupling /= len(all_results)
    
    # Print coupling matrix
    print("  Coupling matrix (row=perturbed, col=affected):")
    for i in range(18):
        row_str = f"  L{i:2d}: "
        for j in range(18):
            row_str += f"{agg_coupling[i,j]:6.2f} "
        print(row_str)
    
    # Diagonal vs off-diagonal
    diag = np.diag(agg_coupling)
    off_diag = agg_coupling[~np.eye(18, dtype=bool)]
    print(f"\n  Diagonal (self-coupling): mean={np.mean(diag):.2f}, std={np.std(diag):.2f}")
    print(f"  Off-diagonal (cross-coupling): mean={np.mean(off_diag):.2f}, std={np.std(off_diag):.2f}")
    print(f"  Cross/self ratio: {np.mean(off_diag)/np.mean(diag):.3f}")
    
    # Layer-wise geometry sensitivity
    print("\n" + "="*60)
    print("LAYER-WISE GEOMETRY SENSITIVITY (scale=2.0)")
    print("="*60)
    
    for metric in ['wh_ratio', 'jaw_width', 'cheek_width', 'temple_width']:
        sensitivities = []
        for layer_idx in range(18):
            deltas = []
            for r in all_results:
                # Find scale=2.0 perturbation
                for p in r['layer_perturbations'][layer_idx]['perturbations']:
                    if abs(p['scale'] - 2.0) < 0.01:
                        deltas.append(p['geometry_delta'][metric])
                        break
            if deltas:
                sensitivities.append((layer_idx, np.mean(deltas), np.std(deltas)))
        
        sensitivities.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  {metric}:")
        for layer_idx, mean_delta, std_delta in sensitivities[:5]:
            print(f"    Layer {layer_idx:2d}: delta={mean_delta:+.4f} ± {std_delta:.4f}")
    
    # Save aggregate
    with open(OUTPUT_DIR / "aggregate_cross_layer_coupling.json", 'w') as f:
        json.dump({
            'coupling_matrix': agg_coupling.tolist(),
            'diagonal_mean': float(np.mean(diag)),
            'diagonal_std': float(np.std(diag)),
            'off_diagonal_mean': float(np.mean(off_diag)),
            'off_diagonal_std': float(np.std(off_diag)),
            'cross_self_ratio': float(np.mean(off_diag)/np.mean(diag)),
            'all_results': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    return all_results

if __name__ == '__main__':
    run_layer_perturbation()