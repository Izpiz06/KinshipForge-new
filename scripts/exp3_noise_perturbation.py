"""
Experiment 3: Noise Perturbation Around latent_avg
Test if tiny perturbations of latent_avg produce large geometric shifts.
Determines if latent manifold itself is biased toward widening.
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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/e4e_geometric_bias_research/exp3_noise_perturbation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'images').mkdir(exist_ok=True)

# Noise magnitudes to test (in latent space units)
NOISE_MAGNITUDES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
N_SAMPLES_PER_MAG = 20  # Number of random perturbations per magnitude

def load_models():
    ckp = torch.load('C:/tmp/ckpt/e4e_ffhq_encode.pt', map_location="cpu", weights_only=False)
    encoder = Encoder4Editing(50, "ir_se", 1024).eval().to(DEVICE)
    encoder.load_state_dict(get_keys(ckp, "encoder"), strict=True)
    
    latent_avg = ckp["latent_avg"].unsqueeze(0).to(DEVICE)  # [1, 18, 512]
    
    generator = Generator(1024, 512, 8).to(DEVICE)
    checkpoint = torch.load('C:/tmp/ckpt/stylegan2-ffhq-config-f.pt', map_location="cpu", weights_only=False)
    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    generator.eval()
    
    return encoder, generator, latent_avg

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

def run_experiment3():
    print("="*70)
    print("EXPERIMENT 3: NOISE PERTURBATION AROUND LATENT_AVG")
    print("="*70)
    
    encoder, generator, latent_avg = load_models()
    
    all_results = []
    
    for mag in NOISE_MAGNITUDES:
        print(f"\nNoise magnitude: {mag}")
        
        mag_results = []
        
        for sample in range(N_SAMPLES_PER_MAG):
            # Generate random noise in latent space
            noise = torch.randn_like(latent_avg) * mag
            w_perturbed = latent_avg + noise
            
            with torch.no_grad():
                img, _ = generator([w_perturbed], return_latents=True, input_is_latent=True)
            
            img_np = tensor2rgb(img)
            geom = measure_geometry_full(img_np)
            
            if geom is None:
                continue
            
            # Compute displacement from latent_avg baseline
            mag_results.append(geom)
        
        if mag_results:
            # Aggregate statistics
            agg = {}
            for key in ['wh_ratio', 'jaw_width', 'cheek_width', 'face_width', 'face_height', 
                       'chin_nose', 'nose_eye', 'eye_forehead', 'chin_forehead']:
                vals = [r[key] for r in mag_results]
                agg[key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals))
                }
            
            result = {
                'noise_magnitude': float(mag),
                'n_samples': len(mag_results),
                'geometry_stats': agg
            }
            all_results.append(result)
            
            print(f"  mag={mag}: WH={agg['wh_ratio']['mean']:.4f}±{agg['wh_ratio']['std']:.4f}, "
                  f"Jaw={agg['jaw_width']['mean']:.1f}±{agg['jaw_width']['std']:.1f}, "
                  f"Cheek={agg['cheek_width']['mean']:.1f}±{agg['cheek_width']['std']:.1f}")
        
        # Save sample images for key magnitudes
        if mag in [0.0, 1.0, 5.0, 10.0, 50.0, 100.0]:
            # Save a representative sample
            for sample in range(min(3, N_SAMPLES_PER_MAG)):
                noise = torch.randn_like(latent_avg) * mag
                w_perturbed = latent_avg + noise
                with torch.no_grad():
                    img, _ = generator([w_perturbed], return_latents=True, input_is_latent=True)
                img_np = tensor2rgb(img)
                cv2.imwrite(str(OUTPUT_DIR / 'images' / f'mag{mag:.1f}_sample{sample}.png'), 
                           cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    
    # Save results
    with open(OUTPUT_DIR / 'exp3_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    generate_plots(all_results)
    print(f"\nResults saved to {OUTPUT_DIR}")

def generate_plots(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    mags = [r['noise_magnitude'] for r in results]
    
    metrics = [
        ('wh_ratio', 'W/H Ratio'),
        ('jaw_width', 'Jaw Width (px)'),
        ('cheek_width', 'Cheek Width (px)'),
        ('face_width', 'Face Width (px)'),
        ('face_height', 'Face Height (px)'),
        ('chin_nose', 'Chin-Nose (px)'),
        ('nose_eye', 'Nose-Eye (px)'),
        ('eye_forehead', 'Eye-Forehead (px)'),
    ]
    
    for key, name in metrics:
        means = [r['geometry_stats'][key]['mean'] for r in results]
        stds = [r['geometry_stats'][key]['std'] for r in results]
        mins = [r['geometry_stats'][key]['min'] for r in results]
        maxs = [r['geometry_stats'][key]['max'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(mags, means, yerr=stds, fmt='o-', capsize=4, label='Mean ± Std')
        plt.fill_between(mags, mins, maxs, alpha=0.2, label='Min-Max Range')
        plt.xscale('log')
        plt.xlabel('Noise Magnitude (log scale)')
        plt.ylabel(name)
        plt.title(f'{name} vs Noise Magnitude Around latent_avg')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'exp3_{key}_vs_noise.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Aggregate plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (key, name) in enumerate(metrics):
        ax = axes[idx]
        means = [r['geometry_stats'][key]['mean'] for r in results]
        stds = [r['geometry_stats'][key]['std'] for r in results]
        ax.errorbar(mags, means, yerr=stds, fmt='o-', capsize=3)
        ax.set_xscale('log')
        ax.set_xlabel('Noise Magnitude')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Geometry Statistics vs Noise Magnitude Around latent_avg')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'exp3_aggregate.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    run_experiment3()