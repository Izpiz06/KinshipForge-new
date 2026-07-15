"""
Task 5: Geometry Preservation Measurement
Measure exact geometry changes through the full pipeline:
Original -> W+ -> W2Sub -> Regional Latents -> Crossover -> Sub2W -> W+ -> Generator
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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/w2sub_sub2w_invertibility/geometry_preservation')
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'intermediate_images').mkdir(exist_ok=True)
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

def difference_heatmap(img1, img2, save_path):
    # Resize to match
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
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

def run_geometry_preservation():
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
    
    all_results = []
    
    for f_file, m_file, gender, race_f, race_m, pair_name in TEST_PAIRS:
        print(f"\n{'='*60}")
        print(f"Processing {pair_name}...")
        print(f"{'='*60}")
        
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        w18_F, aligned_F = load_and_encode(f_path, encoder, mean_latent, DEVICE)
        w18_M, aligned_M = load_and_encode(m_path, encoder, mean_latent, DEVICE)
        
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
        
        # ============================================================
        # STAGE 1: Original parent images (RGB)
        # ============================================================
        geom_F_orig = compute_geometry(aligned_F)
        geom_M_orig = compute_geometry(aligned_M)
        
        # ============================================================
        # STAGE 2: e4e inversion -> W+
        # ============================================================
        with torch.no_grad():
            img_F_w, _ = generator([w18_F], return_latents=True, input_is_latent=True)
            img_M_w, _ = generator([w18_M], return_latents=True, input_is_latent=True)
        geom_F_w = compute_geometry(tensor2rgb(img_F_w))
        geom_M_w = compute_geometry(tensor2rgb(img_M_w))
        
        # ============================================================
        # STAGE 3: W2Sub -> Regional Latents (mu, var, z)
        # ============================================================
        with torch.no_grad():
            mu_F, var_F, sub34_F = w2sub34(w18_F)
            mu_M, var_M, sub34_M = w2sub34(w18_M)
        
        # Sub2W roundtrip (no crossover)
        with torch.no_grad():
            w18_F_rt = sub2w(sub34_F)
            w18_M_rt = sub2w(sub34_M)
            img_F_rt, _ = generator([w18_F_rt], return_latents=True, input_is_latent=True)
            img_M_rt, _ = generator([w18_M_rt], return_latents=True, input_is_latent=True)
        geom_F_rt = compute_geometry(tensor2rgb(img_F_rt))
        geom_M_rt = compute_geometry(tensor2rgb(img_M_rt))
        
        # ============================================================
        # STAGE 4: Full pipeline with crossover
        # ============================================================
        w18_child = fuse_latent(
            w2sub34, sub2w, w18_F=w18_F, w18_M=w18_M,
            random_fakes=random_fakes, fixed_gamma=0.05, fixed_eta=0.4, arcs_lambda=0.0,
            child_gender='male', geometry_weight=0.7, texture_weight=0.5
        )
        
        with torch.no_grad():
            img_child, _ = generator([w18_child], return_latents=True, input_is_latent=True)
        geom_child = compute_geometry(tensor2rgb(img_child))
        
        # ============================================================
        # STAGE 5: Sub2W only (from crossover output, before mix)
        # ============================================================
        # Get the sub34 after crossover but before mix
        # Need to trace inside fuse_latent
        from models.stylegene.gene_crossover_mutation import REGION_SENSITIVITY_MAP, face_class, reparameterize
        import random
        
        # Recreate crossover step
        s_map = REGION_SENSITIVITY_MAP
        s_vals = list(s_map.values())
        s_min = min(s_vals) if s_vals else 0.0
        s_max = max(s_vals) if s_vals else 1.0
        s_range = s_max - s_min if s_max != s_min else 1.0
        
        resolved_gammas = {}
        for name in face_class:
            if name == 'background':
                continue
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
        
        # Sub2W from crossover output
        with torch.no_grad():
            w18_after_sub2w = sub2w(new_sub34)
            img_after_sub2w, _ = generator([w18_after_sub2w], return_latents=True, input_is_latent=True)
        geom_after_sub2w = compute_geometry(tensor2rgb(img_after_sub2w))
        
        # After mix
        with torch.no_grad():
            img_child_final, _ = generator([w18_child], return_latents=True, input_is_latent=True)
        geom_child_final = compute_geometry(tensor2rgb(img_child_final))
        
        # Difference heatmaps
        hm_dir = OUTPUT_DIR / 'heatmaps'
        hm_dir.mkdir(exist_ok=True)
        
        # F -> W+ reconstruction
        hm1 = difference_heatmap(aligned_F, tensor2rgb(img_F_w), hm_dir / f"{pair_name}_father_RGB_to_Wplus.png")
        # W+ -> W2Sub -> Sub2W roundtrip
        hm2 = difference_heatmap(tensor2rgb(img_F_w), tensor2rgb(img_F_rt), hm_dir / f"{pair_name}_father_Wplus_roundtrip.png")
        # Crossover -> Sub2W
        hm3 = difference_heatmap(tensor2rgb(img_after_sub2w), tensor2rgb(img_child_final), hm_dir / f"{pair_name}_sub2w_to_final.png")
        # Original W+ -> Child
        hm4 = difference_heatmap(tensor2rgb(img_F_w), tensor2rgb(img_child_final), hm_dir / f"{pair_name}_Wplus_to_child.png")
        
        result = {
            'pair': pair_name,
            'timestamp': datetime.now().isoformat(),
            'geometry': {
                'father_original': geom_F_orig,
                'mother_original': geom_M_orig,
                'father_Wplus': geom_F_w,
                'mother_Wplus': geom_M_w,
                'father_Wplus_roundtrip': geom_F_rt,
                'mother_Wplus_roundtrip': geom_M_rt,
                'after_sub2w': geom_after_sub2w,
                'child_final': geom_child_final,
            },
            'geometry_deltas': {
                'father_RGB_to_Wplus': {k: geom_F_w[k] - geom_F_orig[k] for k in geom_F_orig},
                'mother_RGB_to_Wplus': {k: geom_M_w[k] - geom_M_orig[k] for k in geom_M_orig},
                'father_Wplus_roundtrip': {k: geom_F_rt[k] - geom_F_w[k] for k in geom_F_w},
                'mother_Wplus_roundtrip': {k: geom_M_rt[k] - geom_M_w[k] for k in geom_M_w},
                'Wplus_to_after_sub2w': {k: geom_after_sub2w[k] - geom_F_w[k] for k in geom_F_w},
                'after_sub2w_to_final': {k: geom_child_final[k] - geom_after_sub2w[k] for k in geom_after_sub2w},
                'Wplus_to_final': {k: geom_child_final[k] - geom_F_w[k] for k in geom_F_w},
            },
            'heatmap_metrics': {
                'father_RGB_to_Wplus': hm1,
                'father_Wplus_roundtrip': hm2,
                'sub2w_to_final': hm3,
                'Wplus_to_child': hm4,
            },
        }
        all_results.append(result)
        
        # Print summary
        print(f"  Father RGB->W+: wh_ratio {geom_F_orig['wh_ratio']:.4f} -> {geom_F_w['wh_ratio']:.4f} (d={geom_F_w['wh_ratio']-geom_F_orig['wh_ratio']:+.4f})")
        print(f"  Father W+ roundtrip: {geom_F_w['wh_ratio']:.4f} -> {geom_F_rt['wh_ratio']:.4f} (d={geom_F_rt['wh_ratio']-geom_F_w['wh_ratio']:+.4f})")
        print(f"  W+ -> after Sub2W: {geom_F_w['wh_ratio']:.4f} -> {geom_after_sub2w['wh_ratio']:.4f} (d={geom_after_sub2w['wh_ratio']-geom_F_w['wh_ratio']:+.4f})")
        print(f"  Sub2W -> Final: {geom_after_sub2w['wh_ratio']:.4f} -> {geom_child_final['wh_ratio']:.4f} (d={geom_child_final['wh_ratio']-geom_after_sub2w['wh_ratio']:+.4f})")
        print(f"  Total W+ -> Child: {geom_F_w['wh_ratio']:.4f} -> {geom_child_final['wh_ratio']:.4f} (d={geom_child_final['wh_ratio']-geom_F_w['wh_ratio']:+.4f})")
        print(f"  Heatmap RGB->W+: mean={hm1['mean_abs_diff']:.1f}, max={hm1['max_abs_diff']:.1f}")
        print(f"  Heatmap W+ roundtrip: mean={hm2['mean_abs_diff']:.1f}, max={hm2['max_abs_diff']:.1f}")
        print(f"  Heatmap Sub2W->Final: mean={hm3['mean_abs_diff']:.1f}, max={hm3['max_abs_diff']:.1f}")
        print(f"  Heatmap W+->Child: mean={hm4['mean_abs_diff']:.1f}, max={hm4['max_abs_diff']:.1f}")
    
    # Aggregate
    print("\n" + "="*60)
    print("AGGREGATE GEOMETRY PRESERVATION STATISTICS")
    print("="*60)
    
    stages = [
        ('father_RGB_to_Wplus', 'e4e Inversion'),
        ('father_Wplus_roundtrip', 'W2Sub->Sub2W Roundtrip'),
        ('Wplus_to_after_sub2w', 'Crossover->Sub2W'),
        ('after_sub2w_to_final', 'Mix Function'),
        ('Wplus_to_final', 'Total Pipeline'),
    ]
    
    for metric in ['wh_ratio', 'jaw_width', 'cheek_width', 'temple_width']:
        for key, label in stages:
            deltas = [r['geometry_deltas'][key][metric] for r in all_results if r['geometry_deltas'][key][metric] != 0]
            if deltas:
                t_stat, p_val = stats.ttest_1samp(deltas, 0)
                d = np.mean(deltas) / np.std(deltas) if np.std(deltas) > 0 else 0
                print(f"  {label} - {metric}: mean={np.mean(deltas):+.4f}, std={np.std(deltas):.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")
    
    # Heatmap aggregates
    for key, label in [
        ('father_RGB_to_Wplus', 'RGB->W+'),
        ('father_Wplus_roundtrip', 'W+ Roundtrip'),
        ('sub2w_to_final', 'Sub2W->Final'),
        ('Wplus_to_child', 'W+->Child'),
    ]:
        means = [r['heatmap_metrics'][key]['mean_abs_diff'] for r in all_results]
        maxs = [r['heatmap_metrics'][key]['max_abs_diff'] for r in all_results]
        print(f"  {label} heatmap: mean={np.mean(means):.1f}±{np.std(means):.1f}, max={np.mean(maxs):.1f}±{np.std(maxs):.1f}")
    
    # Save
    with open(OUTPUT_DIR / 'geometry_preservation.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'all_results': all_results,
        }, f, indent=2)
    
    # CSV
    import csv
    with open(OUTPUT_DIR / 'geometry.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pair', 'stage', 'wh_ratio', 'jaw_width', 'cheek_width', 'temple_width'])
        for r in all_results:
            for stage_name, geom in r['geometry'].items():
                writer.writerow([r['pair'], stage_name, geom['wh_ratio'], geom['jaw_width'], geom['cheek_width'], geom['temple_width']])
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    return all_results

if __name__ == '__main__':
    run_geometry_preservation()