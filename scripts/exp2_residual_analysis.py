"""
Experiment 2: Residual Manipulation
Fixed latent_avg, vary residual magnitude.
Test H1: Encoder residual is the dominant source of widening.
"""

import os
import sys
import json
import torch
import numpy as np
import cv2
import tempfile
from pathlib import Path

sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegan2.model import Generator
from models.encoders.psp_encoders import Encoder4Editing
from models.stylegene.util import get_keys, load_img
from configs import path_ckpt_e4e, path_ckpt_stylegan2
from scripts.legacy.geometry_utils import GeometryEstimator
from preprocess.align_images import align_face

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

SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/e4e_geometric_bias_research/exp2_residual_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'images').mkdir(exist_ok=True)

def load_models():
    ckp = torch.load(path_ckpt_e4e, map_location="cpu", weights_only=False)
    encoder = Encoder4Editing(50, "ir_se", 1024).eval().to(DEVICE)
    encoder.load_state_dict(get_keys(ckp, "encoder"), strict=True)
    
    latent_avg = ckp["latent_avg"].unsqueeze(0).to(DEVICE)  # [1, 18, 512]
    
    generator = Generator(1024, 512, 8).to(DEVICE)
    checkpoint = torch.load(path_ckpt_stylegan2, map_location="cpu", weights_only=False)
    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    generator.eval()
    
    return encoder, generator, latent_avg

def load_and_encode(image_path, encoder, latent_avg):
    raw = cv2.imread(image_path)
    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    aligned = align_face(raw_rgb)
    
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    img_t = load_img(tmp.name).to(DEVICE)
    os.unlink(tmp.name)
    
    with torch.no_grad():
        residual = encoder(torch.nn.functional.interpolate(img_t, size=(256, 256)))
        w18 = residual + latent_avg
    
    return w18, aligned, residual

def tensor2rgb(tensor):
    tensor = (tensor * 0.5 + 0.5) * 255
    tensor = torch.clip(tensor, 0, 255).squeeze(0)
    return tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

def get_landmarks(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    rects = geom_estimator.detector(gray, 1)
    if len(rects) == 0:
        return None
    shape = geom_estimator.predictor(gray, rects[0])
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)

def measure_geometry_full(img_np):
    landmarks = get_landmarks(img_np)
    if landmarks is None:
        return None
    
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    face_width = dist(landmarks[0], landmarks[16])
    face_height = dist(landmarks[8], landmarks[27])
    wh_ratio = face_width / face_height if face_height > 0 else 0
    jaw_width = dist(landmarks[4], landmarks[12])
    cheek_width = dist(landmarks[2], landmarks[14])
    temple_width = dist(landmarks[1], landmarks[15])
    chin_width = dist(landmarks[6], landmarks[10])
    forehead_width = temple_width
    
    nose_tip = landmarks[30]
    left_eye = (landmarks[36] + landmarks[39]) / 2
    right_eye = (landmarks[42] + landmarks[45]) / 2
    eye_center = (left_eye + right_eye) / 2
    
    chin_nose = dist(landmarks[8], nose_tip)
    nose_eye = dist(nose_tip, eye_center)
    eye_forehead = dist(eye_center, landmarks[27])
    chin_forehead = dist(landmarks[8], landmarks[27])
    nose_mouth = dist(landmarks[30], landmarks[51])
    mouth_chin = dist(landmarks[57], landmarks[8])
    interocular_height = abs(left_eye[1] - right_eye[1])
    face_bbox_h = landmarks[:, 1].max() - landmarks[:, 1].min()
    
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
        'face_bbox_height': float(face_bbox_h),
        'landmarks': landmarks.tolist()
    }

def run_experiment2():
    print("="*70)
    print("EXPERIMENT 2: RESIDUAL MANIPULATION")
    print("="*70)
    
    encoder, generator, latent_avg = load_models()
    
    all_results = []
    
    for f_file, m_file, pair_name in TEST_PAIRS:
        print(f"\n{'='*50}")
        print(f"Processing {pair_name}...")
        print(f"{'='*50}")
        
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        for role, img_path in [("father", f_path), ("mother", m_path)]:
            print(f"\n  {role}:")
            
            # Get original encoding
            w18_orig, aligned, residual = load_and_encode(img_path, encoder, latent_avg)
            geom_orig = measure_geometry_full(aligned)
            orig_np = aligned
            
            # Test different residual scales
            for scale in SCALES:
                # w18 = latent_avg + scale * residual
                w18_scaled = latent_avg + scale * residual
                
                with torch.no_grad():
                    img_recon, _ = generator([w18_scaled], return_latents=True, input_is_latent=True)
                
                recon_np = tensor2rgb(img_recon)
                geom_recon = measure_geometry_full(recon_np)
                
                if geom_recon is None:
                    print(f"    scale={scale:.2f}: No face detected!")
                    continue
                
                # Landmark displacement
                lm_orig = np.array(geom_orig['landmarks'])
                lm_recon = np.array(geom_recon['landmarks'])
                disp = np.mean(np.linalg.norm(lm_recon - lm_orig, axis=1))
                
                result = {
                    'pair': pair_name,
                    'role': role,
                    'scale': float(scale),
                    'geometry': geom_recon,
                    'landmark_disp': float(disp)
                }
                all_results.append(result)
                
                print(f"    scale={scale:.2f}: WH={geom_recon['wh_ratio']:.4f}, "
                      f"Jaw={geom_recon['jaw_width']:.1f}, Cheek={geom_recon['cheek_width']:.1f}, "
                      f"Disp={disp:.2f}")
                
                # Save key images
                if scale in [0.0, 0.5, 1.0, 1.5, 2.0]:
                    cv2.imwrite(str(OUTPUT_DIR / 'images' / f'{pair_name}_{role}_scale{scale:.2f}.png'),
                               cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR))
    
    # Save results
    with open(OUTPUT_DIR / 'exp2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    generate_plots(all_results)
    print(f"\nResults saved to {OUTPUT_DIR}")

def generate_plots(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r['pair'], r['role'])].append(r)
    
    metrics = [
        ('wh_ratio', 'W/H Ratio'),
        ('jaw_width', 'Jaw Width (px)'),
        ('cheek_width', 'Cheek Width (px)'),
        ('face_width', 'Face Width (px)'),
        ('face_height', 'Face Height (px)'),
        ('chin_nose', 'Chin-Nose (px)'),
        ('nose_eye', 'Nose-Eye (px)'),
        ('eye_forehead', 'Eye-Forehead (px)'),
        ('landmark_disp', 'Mean Landmark Disp (px)')
    ]
    
    for key, name in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, ((pair, role), group) in enumerate(grouped.items()):
            if idx >= 5:
                break
            ax = axes[idx]
            
            scales = [r['scale'] for r in group]
            if key == 'landmark_disp':
                vals = [r[key] for r in group]
            else:
                vals = [r['geometry'][key] for r in group]
            
            ax.plot(scales, vals, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Residual Scale')
            ax.set_ylabel(name)
            ax.set_title(f'{pair} - {role}')
            ax.grid(True, alpha=0.3)
        
        if len(grouped) < 6:
            axes[5].set_visible(False)
        
        plt.suptitle(f'{name} vs Residual Scale')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'exp2_{key}_vs_scale.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Aggregate plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (key, name) in enumerate(metrics):
        ax = axes[idx]
        
        scales = SCALES
        means = []
        stds = []
        for scale in scales:
            vals = []
            for r in results:
                if abs(r['scale'] - scale) < 0.001:
                    if key == 'landmark_disp':
                        vals.append(r[key])
                    else:
                        vals.append(r['geometry'][key])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        
        ax.errorbar(scales, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, markersize=6)
        ax.set_xlabel('Residual Scale')
        ax.set_ylabel(name)
        ax.set_title(f'Aggregate: {name}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Aggregate Geometry vs Residual Scale')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'exp2_aggregate.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    run_experiment2()