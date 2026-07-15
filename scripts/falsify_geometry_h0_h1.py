"""
Falsification Experiment: H0 vs H1 - Is "Fattening" Horizontal Expansion or Vertical Compression?

H0: Increase in W/H ratio is primarily caused by INCREASE in facial width
H1: Increase in W/H ratio is primarily caused by DECREASE in facial height

Pipeline Stages to Instrument:
1. Original Parent (aligned face)
2. e4e W+ (inversion + reconstruction)
3. W2Sub decomposition
4. Crossover (regional interpolation)
5. Mutation
6. Sub2W reconstruction
7. Mix (final W+)
8. Generator (final child)

At EVERY stage measure:
- Face Width (L0-L16)
- Face Height (L8-L27)
- W/H Ratio
- Jaw Width (L4-L12)
- Cheek Width (L2-L14)
- Temple Width (L0-L16 at temple level)
- Chin Width (L6-L10)
- Forehead Width (L1-L15 at brow level)
- Chin-Nose distance
- Nose-Eye distance
- Eye-Forehead distance
- Chin-Forehead distance
- Nose-Mouth distance
- Mouth-Chin distance
- Interocular Height
- Face Bounding Box Height
- Facial Convex Hull Height

Also compute Δ between consecutive stages.
Landmark displacement vectors (Δx, Δy) per landmark per stage.
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
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegene.api import init_model, tensor2rgb, load_img
from models.stylegene.gene_crossover_mutation import fuse_latent, REGION_SENSITIVITY_MAP, face_class, reparameterize
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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/w2sub_sub2w_invertibility/h0_h1_falsification')
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / 'intermediate_images').mkdir(exist_ok=True)
(OUTPUT_DIR / 'intermediate_latents').mkdir(exist_ok=True)
(OUTPUT_DIR / 'landmarks').mkdir(exist_ok=True)
(OUTPUT_DIR / 'plots').mkdir(exist_ok=True)

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_landmarks(img_np):
    """Get 68 dlib landmarks from image."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    rects = geom_estimator.detector(gray, 1)
    if len(rects) == 0:
        return None
    shape = geom_estimator.predictor(gray, rects[0])
    landmarks = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)
    return landmarks

def compute_comprehensive_geometry(landmarks):
    """Compute all geometric measurements from 68 landmarks."""
    if landmarks is None:
        return None
    
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    # Face width (L0 to L16)
    face_width = dist(landmarks[0], landmarks[16])
    
    # Face height (L8 to L27) - chin to brow
    face_height = dist(landmarks[8], landmarks[27])
    
    # W/H ratio
    wh_ratio = face_width / face_height if face_height > 0 else 0.0
    
    # Jaw width (L4 to L12)
    jaw_width = dist(landmarks[4], landmarks[12])
    
    # Cheek width (L2 to L14)
    cheek_width = dist(landmarks[2], landmarks[14])
    
    # Temple width - approximate using L0 and L16 at temple height (L19/L24)
    # Temple is roughly at eyebrow level
    temple_width = dist(landmarks[1], landmarks[15])
    
    # Chin width (L6 to L10)
    chin_width = dist(landmarks[6], landmarks[10])
    
    # Forehead width (L1 to L15 at brow level)
    forehead_width = dist(landmarks[1], landmarks[15])
    
    # Vertical measurements
    # Chin to nose (L8 to L30 - nose tip)
    chin_nose = dist(landmarks[8], landmarks[30])
    
    # Nose to eye (L30 to midpoint of eyes)
    nose_tip = landmarks[30]
    left_eye_center = (landmarks[36] + landmarks[39]) / 2
    right_eye_center = (landmarks[42] + landmarks[45]) / 2
    eye_center = (left_eye_center + right_eye_center) / 2
    nose_eye = dist(nose_tip, eye_center)
    
    # Eye to forehead (eye center to L27 - brow)
    eye_forehead = dist(eye_center, landmarks[27])
    
    # Chin to forehead (L8 to L27)
    chin_forehead = dist(landmarks[8], landmarks[27])
    
    # Nose to mouth (L30 to L51 - upper lip)
    nose_mouth = dist(landmarks[30], landmarks[51])
    
    # Mouth to chin (L57 - lower lip to L8 - chin)
    mouth_chin = dist(landmarks[57], landmarks[8])
    
    # Interocular height (vertical distance between eyes)
    interocular_height = abs(left_eye_center[1] - right_eye_center[1])
    
    # Face bounding box height
    y_coords = landmarks[:, 1]
    face_bbox_height = y_coords.max() - y_coords.min()
    
    # Facial convex hull height
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(landmarks)
        hull_points = landmarks[hull.vertices]
        hull_height = hull_points[:, 1].max() - hull_points[:, 1].min()
    except:
        hull_height = face_bbox_height
    
    return {
        'face_width': float(face_width),
        'face_height': float(face_height),
        'wh_ratio': float(wh_ratio),
        'jaw_width': float(jaw_width),
        'cheek_width': float(cheek_width),
        'temple_width': float(temple_width),
        'chin_width': float(chin_width),
        'forehead_width': float(forehead_width),
        'chin_nose': float(chin_nose),
        'nose_eye': float(nose_eye),
        'eye_forehead': float(eye_forehead),
        'chin_forehead': float(chin_forehead),
        'nose_mouth': float(nose_mouth),
        'mouth_chin': float(mouth_chin),
        'interocular_height': float(interocular_height),
        'face_bbox_height': float(face_bbox_height),
        'convex_hull_height': float(hull_height),
        'landmarks': landmarks.tolist()
    }

def compute_geometry_from_image(img_np):
    """Compute geometry from image array."""
    landmarks = get_landmarks(img_np)
    if landmarks is None:
        return None
    return compute_comprehensive_geometry(landmarks)

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

def save_image(img_tensor, path):
    img_np = tensor2rgb(img_tensor)
    Image.fromarray(img_np).save(path)

def latent_stats(w18):
    with torch.no_grad():
        w_flat = w18.squeeze(0)
        norms = torch.norm(w_flat, p=2, dim=-1)
        mean_norm = float(norms.mean().item())
        std_norm = float(norms.std().item())
        mean_vec = w_flat.mean(dim=0)
        centered = w_flat - mean_vec
        cov = centered.T @ centered / 17
        trace = float(torch.trace(cov).item())
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        eff_rank = float((eigvals.sum() ** 2 / (eigvals ** 2).sum()).item()) if len(eigvals) > 0 else 0.0
    return {
        'mean_norm': mean_norm, 'std_norm': std_norm,
        'cov_trace': trace, 'effective_rank': eff_rank,
        'eigenvalues': eigvals.tolist()
    }

def compute_delta(dict1, dict2, keys=None):
    """Compute delta between two geometry dicts."""
    if dict1 is None or dict2 is None:
        return {}
    if keys is None:
        keys = [k for k in dict1.keys() if k != 'landmarks']
    return {k: dict2.get(k, 0) - dict1.get(k, 0) for k in keys}

def save_landmarks_json(landmarks, path):
    """Save landmarks to JSON."""
    if landmarks is not None:
        with open(path, 'w') as f:
            json.dump({'landmarks': landmarks.tolist()}, f)

def plot_landmark_overlay(landmarks_list, labels, colors, title, save_path):
    """Plot landmark overlays for multiple stages."""
    fig, ax = plt.subplots(figsize=(10, 10))
    for lm, label, color in zip(landmarks_list, labels, colors):
        if lm is not None:
            ax.scatter(lm[:, 0], lm[:, 1], c=color, label=label, s=20, alpha=0.7)
            # Draw face outline
            outline_idx = list(range(17))  # jawline
            ax.plot(lm[outline_idx, 0], lm[outline_idx, 1], c=color, alpha=0.5, linewidth=1)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(title)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_displacement_vectors(landmarks_prev, landmarks_curr, title, save_path, scale=3.0):
    """Plot displacement vectors between two landmark sets."""
    if landmarks_prev is None or landmarks_curr is None:
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    dx = landmarks_curr[:, 0] - landmarks_prev[:, 0]
    dy = landmarks_curr[:, 1] - landmarks_prev[:, 1]
    # Quiver plot
    ax.quiver(landmarks_prev[:, 0], landmarks_prev[:, 1], dx, dy, 
              angles='xy', scale_units='xy', scale=1/scale, 
              color='red', alpha=0.7, width=0.003, headwidth=3)
    # Original landmarks
    ax.scatter(landmarks_prev[:, 0], landmarks_prev[:, 1], c='blue', s=15, alpha=0.5, label='Previous')
    ax.scatter(landmarks_curr[:, 0], landmarks_curr[:, 1], c='red', s=15, alpha=0.5, label='Current')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(title)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_heatmap_displacement(landmarks_prev, landmarks_curr, title, save_path):
    """Plot vertical/horizontal displacement heatmaps."""
    if landmarks_prev is None or landmarks_curr is None:
        return
    dx = landmarks_curr[:, 0] - landmarks_prev[:, 0]
    dy = landmarks_curr[:, 1] - landmarks_prev[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Horizontal displacement
    scatter = axes[0].scatter(landmarks_prev[:, 0], landmarks_prev[:, 1], 
                              c=dx, cmap='RdBu_r', s=100, vmin=-max(abs(dx)), vmax=max(abs(dx)))
    axes[0].invert_yaxis()
    axes[0].set_aspect('equal')
    axes[0].set_title('Horizontal Displacement (Delta x)')
    plt.colorbar(scatter, ax=axes[0])
    
    # Vertical displacement
    scatter = axes[1].scatter(landmarks_prev[:, 0], landmarks_prev[:, 1], 
                              c=dy, cmap='RdBu_r', s=100, vmin=-max(abs(dy)), vmax=max(abs(dy)))
    axes[1].invert_yaxis()
    axes[1].set_aspect('equal')
    axes[1].set_title('Vertical Displacement (Delta y)')
    plt.colorbar(scatter, ax=axes[1])
    
    plt.suptitle(title)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_geometry_curves(stages_data, pair_name, save_path):
    """Plot width, height, aspect ratio curves across stages."""
    stage_names = list(stages_data.keys())
    metrics = ['face_width', 'face_height', 'wh_ratio', 'jaw_width', 'cheek_width', 
               'temple_width', 'chin_width', 'forehead_width',
               'chin_nose', 'nose_eye', 'eye_forehead', 'chin_forehead',
               'nose_mouth', 'mouth_chin', 'interocular_height',
               'face_bbox_height', 'convex_hull_height']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if idx >= len(axes):
            break
        ax = axes[idx]
        values = [stages_data[s].get(metric, 0) for s in stage_names]
        ax.plot(range(len(stage_names)), values, 'o-', linewidth=2, markersize=8)
        ax.set_xticks(range(len(stage_names)))
        ax.set_xticklabels(stage_names, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} across stages')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Geometry Curves - {pair_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_trajectory(landmarks_list, stage_names, keypoint_idx, label, save_path):
    """Plot trajectory of a specific keypoint across stages."""
    if not landmarks_list:
        return
    xs = [lm[keypoint_idx, 0] if lm is not None else np.nan for lm in landmarks_list]
    ys = [lm[keypoint_idx, 1] if lm is not None else np.nan for lm in landmarks_list]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # X trajectory
    axes[0].plot(range(len(stage_names)), xs, 'o-', linewidth=2, markersize=8)
    axes[0].set_xticks(range(len(stage_names)))
    axes[0].set_xticklabels(stage_names, rotation=45, ha='right')
    axes[0].set_ylabel('X coordinate')
    axes[0].set_title(f'{label} X trajectory')
    axes[0].grid(True, alpha=0.3)
    
    # Y trajectory
    axes[1].plot(range(len(stage_names)), ys, 'o-', linewidth=2, markersize=8)
    axes[1].set_xticks(range(len(stage_names)))
    axes[1].set_xticklabels(stage_names, rotation=45, ha='right')
    axes[1].set_ylabel('Y coordinate')
    axes[1].set_title(f'{label} Y trajectory')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Keypoint {label} (idx={keypoint_idx}) Trajectory')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_h0_h1_falsification():
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
    csv_rows = []
    
    for f_file, m_file, gender, race_f, race_m, pair_name in TEST_PAIRS:
        print(f"\n{'='*60}")
        print(f"Processing {pair_name}...")
        print(f"{'='*60}")
        
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        # ========================
        # STAGE 0: Original Parents
        # ========================
        w18_F, aligned_F = load_and_encode(f_path, encoder, mean_latent, DEVICE)
        w18_M, aligned_M = load_and_encode(m_path, encoder, mean_latent, DEVICE)
        
        lm_F_orig = get_landmarks(aligned_F)
        lm_M_orig = get_landmarks(aligned_M)
        
        geom_F_orig = compute_comprehensive_geometry(lm_F_orig)
        geom_M_orig = compute_comprehensive_geometry(lm_M_orig)
        
        save_landmarks_json(lm_F_orig, OUTPUT_DIR / 'landmarks' / f"{pair_name}_father_stage0_original.json")
        save_landmarks_json(lm_M_orig, OUTPUT_DIR / 'landmarks' / f"{pair_name}_mother_stage0_original.json")
        
        # ========================
        # STAGE 1: e4e W+ Reconstruction
        # ========================
        with torch.no_grad():
            img_F_w, _ = generator([w18_F], return_latents=True, input_is_latent=True)
            img_M_w, _ = generator([w18_M], return_latents=True, input_is_latent=True)
        img_F_w_np = tensor2rgb(img_F_w)
        img_M_w_np = tensor2rgb(img_M_w)
        
        lm_F_w = get_landmarks(img_F_w_np)
        lm_M_w = get_landmarks(img_M_w_np)
        
        geom_F_w = compute_comprehensive_geometry(lm_F_w)
        geom_M_w = compute_comprehensive_geometry(lm_M_w)
        
        save_landmarks_json(lm_F_w, OUTPUT_DIR / 'landmarks' / f"{pair_name}_father_stage1_e4e_wplus.json")
        save_landmarks_json(lm_M_w, OUTPUT_DIR / 'landmarks' / f"{pair_name}_mother_stage1_e4e_wplus.json")
        
        save_image(img_F_w, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_father_Wplus.png")
        save_image(img_M_w, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_mother_Wplus.png")
        
        # ========================
        # STAGE 2: W2Sub Decomposition
        # ========================
        with torch.no_grad():
            mu_F, var_F, sub34_F = w2sub34(w18_F)
            mu_M, var_M, sub34_M = w2sub34(w18_M)
        
        # ========================
        # STAGE 2b: Sub2W Roundtrip (no crossover)
        # ========================
        with torch.no_grad():
            w18_F_rt = sub2w(sub34_F)
            w18_M_rt = sub2w(sub34_M)
            img_F_rt, _ = generator([w18_F_rt], return_latents=True, input_is_latent=True)
            img_M_rt, _ = generator([w18_M_rt], return_latents=True, input_is_latent=True)
        img_F_rt_np = tensor2rgb(img_F_rt)
        img_M_rt_np = tensor2rgb(img_M_rt)
        
        lm_F_rt = get_landmarks(img_F_rt_np)
        lm_M_rt = get_landmarks(img_M_rt_np)
        
        geom_F_rt = compute_comprehensive_geometry(lm_F_rt)
        geom_M_rt = compute_comprehensive_geometry(lm_M_rt)
        
        save_landmarks_json(lm_F_rt, OUTPUT_DIR / 'landmarks' / f"{pair_name}_father_stage2_w2sub_sub2w_roundtrip.json")
        save_landmarks_json(lm_M_rt, OUTPUT_DIR / 'landmarks' / f"{pair_name}_mother_stage2_w2sub_sub2w_roundtrip.json")
        
        save_image(img_F_rt, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_father_roundtrip.png")
        save_image(img_M_rt, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_mother_roundtrip.png")
        
        # ========================
        # STAGE 3: Crossover (full pipeline without mix)
        # ========================
        # ARCS gamma computation
        s_map = REGION_SENSITIVITY_MAP
        s_vals = list(s_map.values())
        s_min = min(s_vals) if s_vals else 0.0
        s_max = max(s_vals) if s_vals else 1.0
        s_range = s_max - s_min if s_max != s_min else 1.0
        
        arcs_gammas = {}
        for name in face_class:
            if name == 'background':
                continue
            s_val = s_map.get(name, 0.0)
            s_norm = (s_val - s_min) / s_range
            g_val = 0.05 * (1.0 - 0.0 * s_norm)
            arcs_gammas[name] = g_val
        
        # BRDAS sampling
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
        
        # Recreate crossover step-by-step for tracking
        with torch.no_grad():
            mu_F, var_F, sub34_F = w2sub34(w18_F)
            mu_M, var_M, sub34_M = w2sub34(w18_M)
        
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
            
            fake_mu, fake_var = random.choice(random_fakes)
            w_i, b_i = weights[classname]
            
            if classname in cur_class:
                new_sub34[:, :, i, :] = reparameterize(
                    mu_F[:, :, i, :] * w_i + fake_mu[:, :, i, :].to(DEVICE) * b_i + mu_M[:, :, i, :] * (1 - w_i - b_i),
                    var_F[:, :, i, :] * w_i + fake_var[:, :, i, :].to(DEVICE) * b_i + var_M[:, :, i, :] * (1 - w_i - b_i))
            else:
                fake_latent = reparameterize(fake_mu[:, :, i, :], fake_var[:, :, i, :]).to(DEVICE)
                new_sub34[:, :, i, :] = new_sub34[:, :, i, :] + fake_latent
        
        with torch.no_grad():
            w18_after_crossover = sub2w(new_sub34)
            img_after_crossover, _ = generator([w18_after_crossover], return_latents=True, input_is_latent=True)
        img_cross_np = tensor2rgb(img_after_crossover)
        
        lm_cross = get_landmarks(img_cross_np)
        geom_cross = compute_comprehensive_geometry(lm_cross)
        
        save_landmarks_json(lm_cross, OUTPUT_DIR / 'landmarks' / f"{pair_name}_child_stage3_crossover.json")
        save_image(img_after_crossover, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_after_crossover.png")
        
        # ========================
        # STAGE 4: Mutation (applied to crossover result)
        # ========================
        w18_after_mutation = fuse_latent(
            w2sub34, sub2w, w18_F=w18_F, w18_M=w18_M,
            random_fakes=random_fakes, fixed_gamma=0.05, fixed_eta=0.4, arcs_lambda=0.0,
            child_gender=gender, geometry_weight=0.7, texture_weight=0.5
        )
        
        with torch.no_grad():
            img_after_mutation, _ = generator([w18_after_mutation], return_latents=True, input_is_latent=True)
        img_mut_np = tensor2rgb(img_after_mutation)
        
        lm_mut = get_landmarks(img_mut_np)
        geom_mut = compute_comprehensive_geometry(lm_mut)
        
        save_landmarks_json(lm_mut, OUTPUT_DIR / 'landmarks' / f"{pair_name}_child_stage4_mutation.json")
        save_image(img_after_mutation, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_after_mutation.png")
        
        # ========================
        # STAGE 5: Mix (final child)
        # ========================
        from models.stylegene.gene_crossover_mutation import mix
        w18_final = mix(w18_F, w18_M, w18_after_mutation, 
                        geometry_weight=0.7, texture_weight=0.5, child_gender=gender)
        
        with torch.no_grad():
            img_final, _ = generator([w18_final], return_latents=True, input_is_latent=True)
        img_final_np = tensor2rgb(img_final)
        
        lm_final = get_landmarks(img_final_np)
        geom_final = compute_comprehensive_geometry(lm_final)
        
        save_landmarks_json(lm_final, OUTPUT_DIR / 'landmarks' / f"{pair_name}_child_stage5_mix.json")
        save_image(img_final, OUTPUT_DIR / 'intermediate_images' / f"{pair_name}_child_final.png")
        
        # ========================
        # COLLECT ALL STAGES FOR THIS PAIR
        # ========================
        # Focus on father's lineage (original -> W+ -> roundtrip -> crossover -> mutation -> mix -> final)
        stages = {
            'stage0_original': {'geometry': geom_F_orig, 'landmarks': lm_F_orig},
            'stage1_e4e_wplus': {'geometry': geom_F_w, 'landmarks': lm_F_w},
            'stage2_w2sub_sub2w_roundtrip': {'geometry': geom_F_rt, 'landmarks': lm_F_rt},
            'stage3_crossover': {'geometry': geom_cross, 'landmarks': lm_cross},
            'stage4_mutation': {'geometry': geom_mut, 'landmarks': lm_mut},
            'stage5_mix': {'geometry': geom_final, 'landmarks': lm_final},
        }
        
        # Compute deltas between consecutive stages
        stage_names = list(stages.keys())
        for i in range(len(stage_names) - 1):
            prev = stage_names[i]
            curr = stage_names[i + 1]
            delta = compute_delta(stages[prev]['geometry'], stages[curr]['geometry'])
            stages[curr]['delta_from_prev'] = delta
            stages[curr]['delta_from_prev']['from_stage'] = prev
            stages[curr]['delta_from_prev']['to_stage'] = curr
        
        # Also compute delta from original for each stage
        for stage_name in stage_names[1:]:
            delta = compute_delta(stages['stage0_original']['geometry'], stages[stage_name]['geometry'])
            stages[stage_name]['delta_from_original'] = delta
        
        # ========================
        # LANDMARK DISPLACEMENT ANALYSIS
        # ========================
        landmark_displacements = {}
        for i in range(len(stage_names) - 1):
            prev_name = stage_names[i]
            curr_name = stage_names[i + 1]
            lm_prev = stages[prev_name]['landmarks']
            lm_curr = stages[curr_name]['landmarks']
            
            if lm_prev is not None and lm_curr is not None:
                dx = lm_curr[:, 0] - lm_prev[:, 0]
                dy = lm_curr[:, 1] - lm_prev[:, 1]
                disp_magnitude = np.sqrt(dx**2 + dy**2)
                
                # Key landmark groups
                landmark_groups = {
                    'jaw': list(range(0, 17)),
                    'chin': [8],
                    'forehead': list(range(17, 27)),
                    'hairline': [19, 20, 21, 22, 23, 24],
                    'cheek': [1, 2, 3, 13, 14, 15],
                    'nose': list(range(27, 36)),
                    'eyes': list(range(36, 48)),
                    'mouth': list(range(48, 68)),
                }
                
                group_stats = {}
                for group_name, indices in landmark_groups.items():
                    if all(idx < len(dx) for idx in indices):
                        group_stats[group_name] = {
                            'mean_dx': float(np.mean(dx[indices])),
                            'mean_dy': float(np.mean(dy[indices])),
                            'mean_disp': float(np.mean(disp_magnitude[indices])),
                            'std_dx': float(np.std(dx[indices])),
                            'std_dy': float(np.std(dy[indices])),
                        }
                
                landmark_displacements[f'{prev_name}_to_{curr_name}'] = {
                    'per_landmark': {
                        'dx': dx.tolist(),
                        'dy': dy.tolist(),
                        'magnitude': disp_magnitude.tolist()
                    },
                    'group_stats': group_stats
                }
        
        # ========================
        # GENERATE VISUALIZATIONS
        # ========================
        # 1. Stage-by-stage overlay
        landmark_list = [stages[s]['landmarks'] for s in stage_names]
        labels = stage_names
        colors = plt.cm.viridis(np.linspace(0, 1, len(stage_names)))
        plot_landmark_overlay(landmark_list, labels, colors, 
                              f'Landmark Overlay - {pair_name} (Father Lineage)',
                              OUTPUT_DIR / 'plots' / f"{pair_name}_landmark_overlay.png")
        
        # 2. Displacement heatmaps for each transition
        for i in range(len(stage_names) - 1):
            prev_name = stage_names[i]
            curr_name = stage_names[i + 1]
            plot_heatmap_displacement(stages[prev_name]['landmarks'], stages[curr_name]['landmarks'],
                                      f'{pair_name}: {prev_name} -> {curr_name}',
                                      OUTPUT_DIR / 'plots' / f"{pair_name}_heatmap_{prev_name}_to_{curr_name}.png")
        
        # 3. Displacement vectors
        for i in range(len(stage_names) - 1):
            prev_name = stage_names[i]
            curr_name = stage_names[i + 1]
            plot_displacement_vectors(stages[prev_name]['landmarks'], stages[curr_name]['landmarks'],
                                      f'{pair_name}: {prev_name} -> {curr_name}',
                                      OUTPUT_DIR / 'plots' / f"{pair_name}_vectors_{prev_name}_to_{curr_name}.png")
        
        # 4. Geometry curves
        stages_geom = {s: stages[s]['geometry'] for s in stage_names}
        plot_geometry_curves(stages_geom, pair_name,
                             OUTPUT_DIR / 'plots' / f"{pair_name}_geometry_curves.png")
        
        # 5. Jaw trajectory (landmark 4 and 12)
        plot_trajectory(landmark_list, stage_names, 4, 'Jaw_Left_L4',
                        OUTPUT_DIR / 'plots' / f"{pair_name}_trajectory_jaw_left.png")
        plot_trajectory(landmark_list, stage_names, 12, 'Jaw_Right_L12',
                        OUTPUT_DIR / 'plots' / f"{pair_name}_trajectory_jaw_right.png")
        
        # 6. Chin trajectory (landmark 8)
        plot_trajectory(landmark_list, stage_names, 8, 'Chin_L8',
                        OUTPUT_DIR / 'plots' / f"{pair_name}_trajectory_chin.png")
        
        # 7. Forehead trajectory (landmark 19, 24)
        plot_trajectory(landmark_list, stage_names, 19, 'Forehead_Left_L19',
                        OUTPUT_DIR / 'plots' / f"{pair_name}_trajectory_forehead_left.png")
        plot_trajectory(landmark_list, stage_names, 24, 'Forehead_Right_L24',
                        OUTPUT_DIR / 'plots' / f"{pair_name}_trajectory_forehead_right.png")
        
        # ========================
        # COMPILE RESULTS
        # ========================
        result = {
            'pair': pair_name,
            'timestamp': datetime.now().isoformat(),
            'stages': {},
            'landmark_displacements': landmark_displacements,
        }
        
        for stage_name in stage_names:
            stage_data = stages[stage_name]
            result['stages'][stage_name] = {
                'geometry': stage_data['geometry'],
                'delta_from_prev': stage_data.get('delta_from_prev', {}),
                'delta_from_original': stage_data.get('delta_from_original', {}),
            }
        
        all_results.append(result)
        
        # Print summary for this pair
        print(f"  Father Original: W={geom_F_orig['face_width']:.1f}, H={geom_F_orig['face_height']:.1f}, WH={geom_F_orig['wh_ratio']:.4f}")
        print(f"  Stage1 W+:       W={geom_F_w['face_width']:.1f}, H={geom_F_w['face_height']:.1f}, WH={geom_F_w['wh_ratio']:.4f}")
        print(f"  Stage2 RT:       W={geom_F_rt['face_width']:.1f}, H={geom_F_rt['face_height']:.1f}, WH={geom_F_rt['wh_ratio']:.4f}")
        print(f"  Stage3 Cross:    W={geom_cross['face_width']:.1f}, H={geom_cross['face_height']:.1f}, WH={geom_cross['wh_ratio']:.4f}")
        print(f"  Stage4 Mut:      W={geom_mut['face_width']:.1f}, H={geom_mut['face_height']:.1f}, WH={geom_mut['wh_ratio']:.4f}")
        print(f"  Stage5 Final:    W={geom_final['face_width']:.1f}, H={geom_final['face_height']:.1f}, WH={geom_final['wh_ratio']:.4f}")
        
        # Add to CSV rows
        for stage_name in stage_names:
            geom = stages[stage_name]['geometry']
            delta_prev = stages[stage_name].get('delta_from_prev', {})
            delta_orig = stages[stage_name].get('delta_from_original', {})
            
            row = {
                'pair': pair_name,
                'stage': stage_name,
                'face_width': geom.get('face_width', -1),
                'face_height': geom.get('face_height', -1),
                'wh_ratio': geom.get('wh_ratio', -1),
                'jaw_width': geom.get('jaw_width', -1),
                'cheek_width': geom.get('cheek_width', -1),
                'temple_width': geom.get('temple_width', -1),
                'chin_width': geom.get('chin_width', -1),
                'forehead_width': geom.get('forehead_width', -1),
                'chin_nose': geom.get('chin_nose', -1),
                'nose_eye': geom.get('nose_eye', -1),
                'eye_forehead': geom.get('eye_forehead', -1),
                'chin_forehead': geom.get('chin_forehead', -1),
                'nose_mouth': geom.get('nose_mouth', -1),
                'mouth_chin': geom.get('mouth_chin', -1),
                'interocular_height': geom.get('interocular_height', -1),
                'face_bbox_height': geom.get('face_bbox_height', -1),
                'convex_hull_height': geom.get('convex_hull_height', -1),
            }
            # Add deltas
            for k, v in delta_prev.items():
                if k not in ['from_stage', 'to_stage']:
                    row[f'delta_prev_{k}'] = v
            for k, v in delta_orig.items():
                row[f'delta_orig_{k}'] = v
            csv_rows.append(row)
    
    # ========================
    # AGGREGATE ANALYSIS
    # ========================
    print("\n" + "="*60)
    print("AGGREGATE ANALYSIS - H0 vs H1 FALSIFICATION")
    print("="*60)
    
    # Key metrics for H0 vs H1
    key_metrics = ['face_width', 'face_height', 'wh_ratio', 'jaw_width', 'cheek_width', 
                   'chin_nose', 'nose_eye', 'eye_forehead', 'chin_forehead',
                   'nose_mouth', 'mouth_chin', 'interocular_height',
                   'face_bbox_height', 'convex_hull_height']
    
    stage_names = ['stage0_original', 'stage1_e4e_wplus', 'stage2_w2sub_sub2w_roundtrip', 
                   'stage3_crossover', 'stage4_mutation', 'stage5_mix']
    
    # Compute aggregate deltas between consecutive stages
    print("\n--- DELTAS BETWEEN CONSECUTIVE STAGES ---")
    for i in range(len(stage_names) - 1):
        prev = stage_names[i]
        curr = stage_names[i + 1]
        print(f"\n{prev} -> {curr}:")
        
        for metric in key_metrics:
            deltas = []
            for r in all_results:
                delta = r['stages'][curr].get('delta_from_prev', {}).get(metric, None)
                if delta is not None and delta != 0:
                    deltas.append(delta)
            
            if deltas:
                mean_d = np.mean(deltas)
                std_d = np.std(deltas)
                t_stat, p_val = stats.ttest_1samp(deltas, 0)
                d = mean_d / std_d if std_d > 0 else 0
                ci = 1.96 * std_d / np.sqrt(len(deltas))
                print(f"  {metric}: Delta={mean_d:+.4f} +/- {std_d:.4f}, n={len(deltas)}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}, CI95=[{mean_d-ci:.4f}, {mean_d+ci:.4f}]")
    
    # Total pipeline delta (original -> final)
    print("\n--- TOTAL PIPELINE (Original -> Final Child) ---")
    for metric in key_metrics:
        deltas = []
        for r in all_results:
            delta = r['stages']['stage5_mix'].get('delta_from_original', {}).get(metric, None)
            if delta is not None and delta != 0:
                deltas.append(delta)
        
        if deltas:
            mean_d = np.mean(deltas)
            std_d = np.std(deltas)
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            d = mean_d / std_d if std_d > 0 else 0
            ci = 1.96 * std_d / np.sqrt(len(deltas))
            print(f"  {metric}: Delta={mean_d:+.4f} +/- {std_d:.4f}, n={len(deltas)}, t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}, CI95=[{mean_d-ci:.4f}, {mean_d+ci:.4f}]")
    
    # ========================
    # H0 vs H1 CORRELATION ANALYSIS
    # ========================
    print("\n--- CORRELATION ANALYSIS: Delta W/H vs Delta Width vs Delta Height ---")
    
    # Collect paired data across all pairs and stages
    delta_wh_all = []
    delta_width_all = []
    delta_height_all = []
    
    for r in all_results:
        for i in range(len(stage_names) - 1):
            prev = stage_names[i]
            curr = stage_names[i + 1]
            d_wh = r['stages'][curr].get('delta_from_prev', {}).get('wh_ratio', None)
            d_w = r['stages'][curr].get('delta_from_prev', {}).get('face_width', None)
            d_h = r['stages'][curr].get('delta_from_prev', {}).get('face_height', None)
            if d_wh is not None and d_w is not None and d_h is not None:
                delta_wh_all.append(d_wh)
                delta_width_all.append(d_w)
                delta_height_all.append(d_h)
    
    if len(delta_wh_all) > 2:
        corr_wh_width = np.corrcoef(delta_wh_all, delta_width_all)[0, 1]
        corr_wh_height = np.corrcoef(delta_wh_all, delta_height_all)[0, 1]
        corr_width_height = np.corrcoef(delta_width_all, delta_height_all)[0, 1]
        
        print(f"  Corr(Delta W/H, Delta Width) = {corr_wh_width:.4f}")
        print(f"  Corr(Delta W/H, Delta Height) = {corr_wh_height:.4f}")
        print(f"  Corr(Delta Width, Delta Height) = {corr_width_height:.4f}")
        
        # Test which correlation is stronger
        if abs(corr_wh_width) > abs(corr_wh_height):
            print(f"  -> H0 SUPPORTED: Delta W/H correlates more strongly with Delta Width ({corr_wh_width:.4f} > {corr_wh_height:.4f})")
        else:
            print(f"  -> H1 SUPPORTED: Delta W/H correlates more strongly with Delta Height ({corr_wh_height:.4f} > {corr_wh_width:.4f})")
        
        # Also test total pipeline
        delta_wh_total = []
        delta_width_total = []
        delta_height_total = []
        for r in all_results:
            d_wh = r['stages']['stage5_mix'].get('delta_from_original', {}).get('wh_ratio', None)
            d_w = r['stages']['stage5_mix'].get('delta_from_original', {}).get('face_width', None)
            d_h = r['stages']['stage5_mix'].get('delta_from_original', {}).get('face_height', None)
            if d_wh is not None and d_w is not None and d_h is not None:
                delta_wh_total.append(d_wh)
                delta_width_total.append(d_w)
                delta_height_total.append(d_h)
        
        if len(delta_wh_total) > 2:
            corr_wh_width_t = np.corrcoef(delta_wh_total, delta_width_total)[0, 1]
            corr_wh_height_t = np.corrcoef(delta_wh_total, delta_height_total)[0, 1]
            print(f"\n  Total Pipeline:")
            print(f"  Corr(Delta W/H, Delta Width) = {corr_wh_width_t:.4f}")
            print(f"  Corr(Delta W/H, Delta Height) = {corr_wh_height_t:.4f}")
            if abs(corr_wh_width_t) > abs(corr_wh_height_t):
                print(f"  -> H0 SUPPORTED")
            else:
                print(f"  -> H1 SUPPORTED")

# ========================
# VERTICAL SEGMENT ANALYSIS
# ========================
    print("\n--- VERTICAL SEGMENT CHANGES (Total Pipeline) ---")
    vertical_metrics = ['chin_nose', 'nose_eye', 'eye_forehead', 'chin_forehead', 'nose_mouth', 'mouth_chin', 
                        'interocular_height', 'face_bbox_height', 'convex_hull_height']
    
    for metric in vertical_metrics:
        deltas = []
        for r in all_results:
            delta = r['stages']['stage5_mix'].get('delta_from_original', {}).get(metric, None)
            if delta is not None and delta != 0:
                deltas.append(delta)
        if deltas:
            mean_d = np.mean(deltas)
            std_d = np.std(deltas)
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            print(f"  {metric}: Delta={mean_d:+.2f} +/- {std_d:.2f}, p={p_val:.4f}")
    
    # ========================
    # SAVE RESULTS
    # ========================
    with open(OUTPUT_DIR / 'h0_h1_falsification_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'hypothesis': 'H0: W/H increase from width increase | H1: W/H increase from height decrease',
            'n_pairs': len(TEST_PAIRS),
            'all_results': all_results,
        }, f, indent=2)
    
    # CSV
    if csv_rows:
        # Collect all possible fieldnames
        all_fieldnames = set()
        for row in csv_rows:
            all_fieldnames.update(row.keys())
        fieldnames = sorted(all_fieldnames)
        with open(OUTPUT_DIR / 'geometry_measurements.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    return all_results

if __name__ == '__main__':
    run_h0_h1_falsification()