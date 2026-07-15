"""
Falsification Test H3: Cross/self Jacobian ≈ 1 implies complete region entanglement
Verify with autograd Jacobian, Integrated Gradients, and sensitivity maps
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from datetime import datetime
from pathlib import Path
from scipy import stats
import tempfile

sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegene.api import init_model, tensor2rgb, load_img
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
]

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/w2sub_sub2w_invertibility/falsification')
OUTPUT_DIR.mkdir(exist_ok=True)

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

def run_falsify_h3():
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
    
    from models.stylegene.data_util import face_class
    
    all_results = []
    
    for f_file, m_file, pair_name in TEST_PAIRS:
        print(f"\nProcessing {pair_name}...")
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        for role, img_path in [("father", f_path), ("mother", m_path)]:
            w18, aligned = load_and_encode(img_path, encoder, mean_latent, DEVICE)
            
            # Get regional latents
            with torch.no_grad():
                mu, var, z = w2sub34(w18)  # z: [1, 18, 34, 512]
            
            # ===== METHOD 1: Autograd Jacobian =====
            # J_ij = d(geometry_metric) / d(z_region_j)
            # We'll use wh_ratio as output
            z_req = z.clone().requires_grad_(True)
            
            # Forward through Sub2W
            w18_recon = sub2w(z_req)
            
            # Generate image
            img_recon, _ = generator([w18_recon], return_latents=True, input_is_latent=True)
            img_np = tensor2rgb(img_recon)
            
            # Compute geometry (differentiable via landmarks would be ideal, 
            # but we'll use a proxy: the latent norm of geometry layers 8-11)
            # For true geometry, we'd need differentiable landmark detection
            # Here we use latent geometry proxy: norm of layers 8-11
            geom_proxy = w18_recon[:, 8:12, :].norm(dim=-1).mean()  # scalar
            
            # Jacobian via autograd
            jac = torch.autograd.grad(geom_proxy, z_req, create_graph=False)[0]  # [1, 18, 34, 512]
            jac = jac.squeeze(0)  # [18, 34, 512]
            
            # Compute cross-region Jacobian: for each region, how much does it affect OTHER regions' latents?
            # Actually we want: how does perturbing region i affect region j's geometry contribution?
            # Use: J_region = d(wh_ratio) / d(z_region)
            
            # Per-region Jacobian norm
            region_jac_norms = {}
            for i, name in enumerate(face_class):
                if name == 'background':
                    continue
                region_jac = jac[:, i, :].norm().item()
                region_jac_norms[name] = region_jac
            
            # ===== METHOD 2: Integrated Gradients =====
            def ig_attr(input_z, target_layer_range=(8, 12), steps=20):
                """Integrated gradients from baseline (zeros) to input_z"""
                baseline = torch.zeros_like(input_z)
                alphas = torch.linspace(0, 1, steps).to(DEVICE)
                ig_sum = torch.zeros_like(input_z)
                
                for alpha in alphas:
                    interp = baseline + alpha * (input_z - baseline)
                    interp.requires_grad_(True)
                    with torch.enable_grad():
                        w18_interp = sub2w(interp)
                        # Geometry proxy: layers 8-11 (geometry layers)
                        proxy = w18_interp[:, target_layer_range[0]:target_layer_range[1], :].norm(dim=-1).mean()
                        grad = torch.autograd.grad(proxy, interp, create_graph=False)[0]
                        ig_sum += grad
                
                ig_attr = (input_z - baseline) * ig_sum / steps
                return ig_sum / steps  # average gradient
            
            ig = ig_attr(z)
            ig_region_norms = {}
            for i, name in enumerate(face_class):
                if name == 'background':
                    continue
                # ig shape: [1, 18, 34, 512] - index 1 is region
                ig_region_norms[name] = ig[:, :, i, :].norm().item()
            
            # ===== METHOD 3: Finite Difference Sensitivity =====
            eps = 0.1
            sens = {}
            for i, name in enumerate(face_class):
                if name == 'background':
                    continue
                z_pert = z.clone()
                z_pert[:, :, i, :] += eps
                
                with torch.no_grad():
                    w18_pert = sub2w(z_pert)
                    img_pert, _ = generator([w18_pert], return_latents=True, input_is_latent=True)
                geom_pert = compute_geometry(tensor2rgb(img_pert))
                
                z_pert_neg = z.clone()
                z_pert_neg[:, :, i, :] -= eps
                with torch.no_grad():
                    w18_pert_neg = sub2w(z_pert_neg)
                    img_pert_neg, _ = generator([w18_pert_neg], return_latents=True, input_is_latent=True)
                geom_neg = compute_geometry(tensor2rgb(img_pert_neg))
                
                sens[name] = {
                    'wh_ratio': (geom_pert['wh_ratio'] - geom_neg['wh_ratio']) / (2 * eps) if geom_pert['wh_ratio'] > 0 and geom_neg['wh_ratio'] > 0 else 0,
                    'jaw_width': (geom_pert['jaw_width'] - geom_neg['jaw_width']) / (2 * eps) if geom_pert['jaw_width'] > 0 and geom_neg['jaw_width'] > 0 else 0,
                    'cheek_width': (geom_pert['cheek_width'] - geom_neg['cheek_width']) / (2 * eps) if geom_pert['cheek_width'] > 0 and geom_neg['cheek_width'] > 0 else 0,
                }
            
            # ===== METHOD 4: Cross/Self Jacobian (autograd on W+) =====
            # d(W+_k) / d(z_region_j) for geometry layers k=8..11
            z_req2 = z.clone().requires_grad_(True)
            w18_jac = sub2w(z_req2)
            
            # Jacobian of geometry layers (8-11) wrt all regions
            jac_w_geom = torch.autograd.grad(
                w18_jac[:, 8:12, :].sum(), z_req2, create_graph=False
            )[0].squeeze(0)  # [18, 34, 512] -> [34, 512] after sum over layers? No, let's do per-layer
            
            # Better: Jacobian of each geometry layer
            cross_self_jac = {}
            for layer_k in range(8, 12):
                z_req_layer = z.clone().requires_grad_(True)
                w18_layer = sub2w(z_req_layer)
                grad = torch.autograd.grad(w18_layer[:, layer_k, :].sum(), z_req_layer, create_graph=False)[0]
                # grad: [1, 18, 34, 512]
                grad = grad.squeeze(0)
                # For each region j, the gradient of layer k wrt region j
                for j, name in enumerate(face_class):
                    if name == 'background':
                        continue
                    cross_self_jac[f'layer_{layer_k}_region_{name}'] = grad[:, j, :].norm().item()
            
            # Compute cross/self ratio
            # Extract diagonal and off-diagonal
            self_vals = []
            cross_vals = []
            for k, v in cross_self_jac.items():
                if 'region_' in k:
                    parts = k.split('_')
                    try:
                        src_idx = int(parts[1])
                        tgt_idx = int(parts[3])
                        if src_idx == tgt_idx:
                            self_vals.append(v)
                        else:
                            cross_vals.append(v)
                    except (ValueError, IndexError):
                        pass
            
            result = {
                'pair': pair_name,
                'role': role,
                'autograd_jacobian': region_jac_norms,
                'integrated_gradients': ig_region_norms,
                'finite_diff_sensitivity': sens,
                'cross_self_jacobian': cross_self_jac,
                'self_jac_mean': float(self_jac),
                'cross_jac_mean': float(cross_jac),
                'cross_self_ratio': float(cross_jac / self_jac) if self_jac > 0 else 0,
            }
            all_results.append(result)
            
            print(f"  {role}: cross/self = {result['cross_self_ratio']:.4f}, self_jac={self_jac:.2f}, cross_jac={cross_jac:.2f}")
            print(f"  Top sens (wh_ratio): {sorted(sens.items(), key=lambda x: abs(x[1]['wh_ratio']), reverse=True)[:3]}")
    
    # Aggregate
    print("\n" + "="*60)
    print("AGGREGATE CROSS/SELF JACOBIAN")
    print("="*60)
    
    ratios = [r['cross_self_ratio'] for r in all_results]
    print(f"Cross/Self ratio: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")
    
    # Falsification
    print("\n" + "="*60)
    print("FALSIFICATION ASSESSMENT - H3: Cross/self ≈ 1 → full entanglement")
    print("="*60)
    
    mean_ratio = np.mean(ratios)
    if mean_ratio > 0.8:
        print(f"  -> Cross/Self = {mean_ratio:.4f} → SUPPORTS full entanglement")
    elif mean_ratio > 0.5:
        print(f"  -> Cross/Self = {mean_ratio:.4f} → PARTIAL entanglement")
    else:
        print(f"  -> Cross/Self = {mean_ratio:.4f} → FAILS entanglement claim")
    
    # Save
    with open(OUTPUT_DIR / 'falsify_H3_jacobian.json', 'w') as f:
        json.dump({
            'hypothesis': 'H3: Cross/self Jacobian ≈ 1 implies complete region entanglement',
            'timestamp': datetime.now().isoformat(),
            'methods': ['autograd_jacobian', 'integrated_gradients', 'finite_diff_sensitivity', 'cross_self_jacobian'],
            'n_samples': len(all_results),
            'results': all_results,
            'falsification': {
                'cross_self_ratio': float(mean_ratio),
                'conclusion': 'PARTIALLY SUPPORTED' if mean_ratio > 0.5 else 'FALSIFIED',
                'note': 'Cross-region influence exists but may be Sub2W mixing, not W2Sub independence failure',
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/falsify_H3_jacobian.json")
    return all_results

if __name__ == '__main__':
    run_falsify_h3()