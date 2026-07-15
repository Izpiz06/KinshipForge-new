"""
Task 4: Region Coupling Analysis
Determine whether supposedly independent regions are actually coupled.
Measure cross-region covariance, cross-region Jacobian, gradient propagation.
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
import tempfile

sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegene.api import init_model, tensor2rgb, load_img
from models.stylegene.gene_crossover_mutation import fuse_latent, REGION_SENSITIVITY_MAP, face_class
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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/w2sub_sub2w_invertibility/region_coupling')
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

def run_region_coupling():
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
    
    # Non-background region indices
    non_bg_regions = [i for i, name in enumerate(face_class) if name != 'background']
    region_names = [face_class[i] for i in non_bg_regions]
    
    all_results = []
    
    for f_file, m_file, gender, race_f, race_m, pair_name in TEST_PAIRS:
        print(f"\nProcessing {pair_name}...")
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
        
        # Get baseline (no perturbation)
        w18_baseline = fuse_latent(
            w2sub34, sub2w, w18_F=w18_F, w18_M=w18_M,
            random_fakes=random_fakes, fixed_gamma=0.05, fixed_eta=0.4, arcs_lambda=0.0,
            child_gender='male', geometry_weight=0.7, texture_weight=0.5
        )
        
        with torch.no_grad():
            img_baseline, _ = generator([w18_baseline], return_latents=True, input_is_latent=True)
        geom_baseline = compute_geometry(tensor2rgb(img_baseline))
        
        # 1. CROSS-REGION COVARIANCE ANALYSIS
        # Get regional latents for baseline
        with torch.no_grad():
            mu_F, var_F, sub34_F = w2sub34(w18_F)
            mu_M, var_M, sub34_M = w2sub34(w18_M)
        
        # Compute cross-region covariance in z-space
        z_F = sub34_F.squeeze(0)  # [18, 34, 512] -> but we want [34, 18*512] for cross-region
        z_M = sub34_M.squeeze(0)
        
        # Reshape: [18, 34, 512] -> [34, 9216] for cross-region covariance
        z_F_flat = z_F.permute(1, 0, 2).reshape(34, -1)  # [34, 18*512]
        z_M_flat = z_M.permute(1, 0, 2).reshape(34, -1)
        
        # Center
        z_F_centered = z_F_flat - z_F_flat.mean(dim=1, keepdim=True)
        z_M_centered = z_M_flat - z_M_flat.mean(dim=1, keepdim=True)
        
        # Cross-region covariance: [34, 34]
        cov_cross_F = (z_F_centered @ z_F_centered.T) / (z_F_flat.shape[1] - 1)
        cov_cross_M = (z_M_centered @ z_M_centered.T) / (z_M_flat.shape[1] - 1)
        
        # 2. REGION PERTURBATION EXPERIMENTS
        # Perturb each region's z and measure effect on other regions and geometry
        region_perturbations = []
        scale = 2.0
        
        for region_idx in non_bg_regions:
            region_name = face_class[region_idx]
            
            # Perturb only this region in father's sub34
            sub34_pert = sub34_F.clone()
            noise = torch.randn_like(sub34_pert[:, :, region_idx, :]) * scale
            sub34_pert[:, :, region_idx, :] += noise
            
            # Pass through Sub2W
            with torch.no_grad():
                w18_pert = sub2w(sub34_pert)
                img_pert, _ = generator([w18_pert], return_latents=True, input_is_latent=True)
            
            geom_pert = compute_geometry(tensor2rgb(img_pert))
            geom_delta = {k: geom_pert[k] - geom_baseline[k] for k in geom_baseline}
            
            # Also measure effect on OTHER regions' z after roundtrip
            with torch.no_grad():
                mu_p, var_p, z_p = w2sub34(w18_pert)
            
            # Compute displacement of OTHER regions
            other_displacements = {}
            for other_idx in non_bg_regions:
                if other_idx != region_idx:
                    disp = float(torch.norm(z_p[:, :, other_idx, :] - sub34_F[:, :, other_idx, :], p=2).item())
                    other_displacements[face_class[other_idx]] = disp
            
            region_perturbations.append({
                'region': region_name,
                'region_idx': region_idx,
                'geometry_delta': geom_delta,
                'other_region_displacements': other_displacements,
            })
            
            print(f"  {region_name}: wh_delta={geom_delta['wh_ratio']:+.4f}, jaw_delta={geom_delta['jaw_width']:+.1f}, cheek_delta={geom_delta['cheek_width']:+.1f}")
        
        # 3. GRADIENT PROPAGATION: Compute Jacobian of output regions wrt input regions
        # Using finite differences
        jacobian = np.zeros((len(non_bg_regions), len(non_bg_regions)))
        eps = 0.1
        
        # Get baseline z for father
        with torch.no_grad():
            _, _, z_baseline = w2sub34(w18_F)
        z_baseline = z_baseline.squeeze(0)  # [18, 34, 512]
        
        for i, src_idx in enumerate(non_bg_regions):
            # Perturb source region
            w18_pert = w18_F.clone()
            # Need to perturb at W2Sub level, not W+ level
            # Perturb sub34_F directly
            sub34_pert = sub34_F.clone()
            sub34_pert[:, :, src_idx, :] += eps
            
            with torch.no_grad():
                w18_recon = sub2w(sub34_pert)
                _, _, z_recon = w2sub34(w18_recon)
            z_recon = z_recon.squeeze(0)
            
            # Measure change in all target regions
            for j, tgt_idx in enumerate(non_bg_regions):
                delta = float(torch.norm(z_recon[:, tgt_idx, :] - z_baseline[:, tgt_idx, :], p=2).item())
                jacobian[i, j] = delta / eps
        
        # 4. GEOMETRY-REGION MAPPING
        # Map which regions most affect geometry metrics
        geometry_sensitivity = {}
        for metric in ['wh_ratio', 'jaw_width', 'cheek_width', 'temple_width', 'face_contour_area']:
            sensitivities = []
            for rp in region_perturbations:
                sensitivities.append((rp['region'], rp['geometry_delta'][metric]))
            sensitivities.sort(key=lambda x: abs(x[1]), reverse=True)
            geometry_sensitivity[metric] = sensitivities[:10]
        
        result = {
            'pair': pair_name,
            'timestamp': datetime.now().isoformat(),
            'baseline_geometry': geom_baseline,
            'cross_region_covariance_father': cov_cross_F.cpu().numpy().tolist(),
            'cross_region_covariance_mother': cov_cross_M.cpu().numpy().tolist(),
            'region_perturbations': region_perturbations,
            'jacobian': jacobian.tolist(),
            'jacobian_region_names': region_names,
            'geometry_sensitivity': geometry_sensitivity,
        }
        all_results.append(result)
        
        # Save per-pair
        with open(OUTPUT_DIR / f"{pair_name}_region_coupling.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    # Aggregate Jacobian
    agg_jacobian = np.zeros_like(all_results[0]['jacobian'])
    for r in all_results:
        agg_jacobian += np.array(r['jacobian'])
    agg_jacobian /= len(all_results)
    
    print("\n" + "="*60)
    print("AGGREGATE REGION COUPLING RESULTS")
    print("="*60)
    
    # Diagonal vs off-diagonal
    diag = np.diag(agg_jacobian)
    off_diag = agg_jacobian[~np.eye(len(non_bg_regions), dtype=bool)]
    print(f"Diagonal (self): mean={np.mean(diag):.4f}, std={np.std(diag):.4f}")
    print(f"Off-diagonal (cross): mean={np.mean(off_diag):.4f}, std={np.std(off_diag):.4f}")
    print(f"Cross/self ratio: {np.mean(off_diag)/np.mean(diag):.4f}")
    
    # Top coupled region pairs
    print("\nTop cross-region couplings:")
    for i in range(len(non_bg_regions)):
        for j in range(len(non_bg_regions)):
            if i != j:
                pass  # Too many pairs
    
    # Geometry sensitivity summary
    print("\nGeometry sensitivity (top 3 regions per metric):")
    for metric in ['wh_ratio', 'jaw_width', 'cheek_width']:
        if metric in all_results[0]['geometry_sensitivity']:
            # Average across pairs
            region_scores = {}
            for r in all_results:
                for region, val in r['geometry_sensitivity'][metric]:
                    if region not in region_scores:
                        region_scores[region] = []
                    region_scores[region].append(val)
            
            avg_scores = {k: np.mean(v) for k, v in region_scores.items()}
            sorted_regions = sorted(avg_scores.items(), key=lambda x: abs(x[1]), reverse=True)
            print(f"  {metric}:")
            for region, val in sorted_regions[:5]:
                print(f"    {region}: {val:+.4f}")
    
    # Save aggregate
    with open(OUTPUT_DIR / "aggregate_region_coupling.json", 'w') as f:
        json.dump({
            'aggregate_jacobian': agg_jacobian.tolist(),
            'diagonal_mean': float(np.mean(diag)),
            'off_diagonal_mean': float(np.mean(off_diag)),
            'cross_self_ratio': float(np.mean(off_diag)/np.mean(diag)),
            'geometry_sensitivity': {
                metric: [
                    {'region': region, 'mean_sensitivity': float(val)}
                    for region, val in sorted(
                        {region: np.mean([r['geometry_sensitivity'][metric][i][1] for r in all_results if i < len(r['geometry_sensitivity'][metric])]) 
                         for i in range(len(all_results[0]['geometry_sensitivity'][metric]))}.items(),
                        key=lambda x: abs(x[1]), reverse=True)
                ]
                for metric in ['wh_ratio', 'jaw_width', 'cheek_width', 'temple_width', 'face_contour_area']
            },
            'all_results': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    return all_results

if __name__ == '__main__':
    run_region_coupling()