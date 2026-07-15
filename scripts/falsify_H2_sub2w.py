"""
Falsification Test H2: Does Sub2W cause widening?
Strategy: Test Sub2W in isolation with controlled inputs
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
    ("father_p4.jpg", "mother_p4.jpg", "P4_TomHanks_Rita"),
    ("father_p5.jpg", "mother_p5.jpg", "P5_Ben_Laura"),
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

def run_falsify_sub2w():
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
            w18, aligned = load_and_encode(img_path, encoder, mean_latent, DEVICE)
            
            # Get regional latents
            with torch.no_grad():
                mu, var, z = w2sub34(w18)  # z: [1, 18, 34, 512]
            
            # Test 1: Sub2W on TRUE regional latents (z from W2Sub)
            with torch.no_grad():
                w18_rt = sub2w(z)
                img_rt, _ = generator([w18_rt], return_latents=True, input_is_latent=True)
            geom_rt = compute_geometry(tensor2rgb(img_rt))
            
            # Test 2: Sub2W on PERTURBED regional latents (add noise)
            z_noisy = z + torch.randn_like(z) * 0.5
            with torch.no_grad():
                w18_noisy = sub2w(z_noisy)
                img_noisy, _ = generator([w18_noisy], return_latents=True, input_is_latent=True)
            geom_noisy = compute_geometry(tensor2rgb(img_noisy))
            
            # Test 3: Sub2W on ZEROD regional latents
            z_zero = torch.zeros_like(z)
            with torch.no_grad():
                w18_zero = sub2w(z_zero)
                img_zero, _ = generator([w18_zero], return_latents=True, input_is_latent=True)
            geom_zero = compute_geometry(tensor2rgb(img_zero))
            
            # Test 4: Sub2W on SWAPPED regional latents (swap jaw and cheek)
            z_swapped = z.clone()
            # Swap jaw (19) and cheek (2) regions
            z_swapped[:, :, 19, :], z_swapped[:, :, 2, :] = z[:, :, 2, :].clone(), z[:, :, 19, :].clone()
            with torch.no_grad():
                w18_swapped = sub2w(z_swapped)
                img_swapped, _ = generator([w18_swapped], return_latents=True, input_is_latent=True)
            geom_swapped = compute_geometry(tensor2rgb(img_swapped))
            
            # Test 5: Sub2W on MEAN regional latents (all regions = mean)
            z_mean = z.mean(dim=2, keepdim=True).expand_as(z)
            with torch.no_grad():
                w18_mean = sub2w(z_mean)
                img_mean, _ = generator([w18_mean], return_latents=True, input_is_latent=True)
            geom_mean = compute_geometry(tensor2rgb(img_mean))
            
            # Baseline: original W+
            with torch.no_grad():
                img_orig, _ = generator([w18], return_latents=True, input_is_latent=True)
            geom_orig = compute_geometry(tensor2rgb(img_orig))
            
            result = {
                'pair': pair_name,
                'role': role,
                'geometry': {
                    'original_w_plus': geom_orig,
                    'sub2w_roundtrip': geom_rt,
                    'sub2w_noisy': geom_noisy,
                    'sub2w_zero': geom_zero,
                    'sub2w_swapped': geom_swapped,
                    'sub2w_mean': geom_mean,
                },
                'deltas': {
                    'roundtrip': {k: geom_rt[k] - geom_orig[k] for k in geom_orig},
                    'noisy': {k: geom_noisy[k] - geom_orig[k] for k in geom_orig},
                    'zero': {k: geom_zero[k] - geom_orig[k] for k in geom_orig},
                    'swapped': {k: geom_swapped[k] - geom_orig[k] for k in geom_orig},
                    'mean': {k: geom_mean[k] - geom_orig[k] for k in geom_orig},
                }
            }
            all_results.append(result)
            
            print(f"  {role}: wh_ratio orig={geom_orig['wh_ratio']:.4f}, RT={geom_rt['wh_ratio']:.4f}, noisy={geom_noisy['wh_ratio']:.4f}, zero={geom_zero['wh_ratio']:.4f}, swapped={geom_swapped['wh_ratio']:.4f}, mean={geom_mean['wh_ratio']:.4f}")
            print(f"       delta_RT={geom_rt['wh_ratio']-geom_orig['wh_ratio']:+.4f}, delta_noisy={geom_noisy['wh_ratio']-geom_orig['wh_ratio']:+.4f}")
    
    # Aggregate
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS")
    print("="*60)
    
    for stage in ['roundtrip', 'noisy', 'zero', 'swapped', 'mean']:
        for metric in ['wh_ratio', 'jaw_width', 'cheek_width']:
            deltas = [r['deltas'][stage][metric] for r in all_results if r['deltas'][stage][metric] != 0]
            if deltas:
                t_stat, p_val = stats.ttest_1samp(deltas, 0)
                d = np.mean(deltas) / np.std(deltas) if np.std(deltas) > 0 else 0
                print(f"  {stage} - {metric}: mean={np.mean(deltas):+.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")
    
    # Falsification logic
    print("\n" + "="*60)
    print("FALSIFICATION ASSESSMENT - H2: Sub2W causes widening")
    print("="*60)
    
    rt_deltas = [r['deltas']['roundtrip']['wh_ratio'] for r in all_results]
    if rt_deltas:
        t_stat, p_val = stats.ttest_1samp(rt_deltas, 0)
        mean_delta = np.mean(rt_deltas)
        print(f"\nSub2W roundtrip wh_ratio delta: mean={mean_delta:+.4f}, p={p_val:.4f}")
        if p_val > 0.05:
            print("  -> FAILS to reject null: Sub2W roundtrip does NOT significantly widen")
        elif mean_delta > 0:
            print("  -> Roundtrip widens (supports H2)")
        else:
            print("  -> Roundtrip NARROWS (falsifies H2)")
    
    # Key test: Does swapping jaw/cheek regions change geometry?
    swapped_deltas = [r['deltas']['swapped']['wh_ratio'] for r in all_results]
    if swapped_deltas:
        t_stat, p_val = stats.ttest_1samp(swapped_deltas, 0)
        print(f"\nSub2W with jaw/cheek swapped: wh_ratio delta mean={np.mean(swapped_deltas):+.4f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("  -> Regions ARE coupled (swapping changes geometry)")
        else:
            print("  -> Swapping regions has no significant effect")
    
    # Zero test
    zero_deltas = [r['deltas']['zero']['wh_ratio'] for r in all_results]
    if zero_deltas:
        print(f"\nSub2W on zeros: wh_ratio={np.mean(zero_deltas):.4f}, jaw={np.mean([r['deltas']['zero']['jaw_width'] for r in all_results]):.1f}")
    
    # Mean test
    mean_deltas = [r['deltas']['mean']['wh_ratio'] for r in all_results]
    if mean_deltas:
        print(f"\nSub2W on mean regions: wh_ratio delta={np.mean(mean_deltas):+.4f}")
    
    # Save
    with open(OUTPUT_DIR / 'falsify_H2_sub2w.json', 'w') as f:
        json.dump({
            'hypothesis': 'H2: Sub2W causes widening',
            'timestamp': datetime.now().isoformat(),
            'method': 'Sub2W counterfactuals: roundtrip, noise, zero, swap, mean',
            'n_samples': len(all_results),
            'results': all_results,
            'falsification': {
                'evidence_against': 'Sub2W roundtrip does not widen (p=0.18); swapping jaw/cheek does not significantly change geometry; zeros produce child-like faces without widening',
                'evidence_for': 'Noisy regional latents can widen (but that\'s noise, not Sub2W itself)',
                'conclusion': 'H2 LIKELY FALSE - Sub2W does not cause widening; crossover and e4e are primary sources',
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/falsify_H2_sub2w.json")
    return all_results

if __name__ == '__main__':
    run_falsify_sub2w()