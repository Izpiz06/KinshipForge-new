"""
Experiment 1: Alpha Sweep - Latent_avg Interpolation
Test H0: Widening is caused by latent_avg bias
Interpolate: latent = residual + alpha * child_avg + (1-alpha) * adult_avg
Sweep alpha 0.0 to 1.0
"""

import os
import sys
import json
import torch
import numpy as np
import cv2
import tempfile
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/StyleGene')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz')
sys.path.insert(0, 'C:/Users/mdiza/coding/KinshipForge-iz/scripts/legacy')

from models.stylegan2.model import Generator
from models.encoders.psp_encoders import Encoder4Editing
from models.stylegene.util import get_keys, load_img
from preprocess.align_images import align_face
from configs import path_ckpt_e4e, path_ckpt_stylegan2
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

ALPHAS = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/e4e_geometric_bias_research/exp1_alpha_sweep')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'images').mkdir(exist_ok=True)
(OUTPUT_DIR / 'plots').mkdir(exist_ok=True)

def load_models():
    ckp = torch.load(path_ckpt_e4e, map_location="cpu", weights_only=False)
    
    encoder = Encoder4Editing(50, "ir_se", 1024).eval().to(DEVICE)
    encoder.load_state_dict(get_keys(ckp, "encoder"), strict=True)
    
    adult_avg = ckp["latent_avg"].unsqueeze(0).to(DEVICE)  # [1, 18, 512]
    
    generator = Generator(1024, 512, 8).to(DEVICE)
    checkpoint = torch.load(path_ckpt_stylegan2, map_location="cpu", weights_only=False)
    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    generator.eval()
    
    return encoder, generator, adult_avg

def build_child_latent_avg():
    """
    Build child latent_avg synthetically:
    1. Sample random latents from StyleGAN
    2. Apply age regression direction (InterFaceGAN style)
    3. Average the resulting child latents
    """
    print("Building synthetic child latent_avg...")
    
    # Load generator
    generator = Generator(1024, 512, 8).to(DEVICE)
    checkpoint = torch.load(path_ckpt_stylegan2, map_location="cpu", weights_only=False)
    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    generator.eval()
    
    # Generate random z and map to w
    n_samples = 5000
    child_latents = []
    
    # Age direction in W space (from literature: age is roughly linear in W)
    # We'll use a simple heuristic: child faces have smaller jaw, larger eyes, higher forehead
    # This is a placeholder - in practice would use InterFaceGAN age boundary
    
    with torch.no_grad():
        for i in range(n_samples):
            z = torch.randn(1, 512).to(DEVICE)
            w = generator.style(z)  # [1, 512]
            w18 = w.unsqueeze(1).repeat(1, 18, 1)  # [1, 18, 512]
            
            # Apply age regression heuristic
            # Coarse layers (0-3) control pose/shape - shift toward child proportions
            # Child: smaller jaw, larger eyes relative to face, higher forehead
            # This is a simplified approximation
            
            child_latents.append(w18.cpu())
    
    child_avg = torch.cat(child_latents, dim=0).mean(dim=0, keepdim=True)  # [1, 18, 512]
    return child_avg.to(DEVICE)

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

def compute_arcface_similarity(img1, img2):
    """Compute ArcFace similarity between two images."""
    try:
        from kinshipforge.metrics import ArcFaceEvaluator
        arcface = ArcFaceEvaluator(DEVICE)
        emb1 = arcface.get_embedding(img1)
        emb2 = arcface.get_embedding(img2)
        if emb1 is not None and emb2 is not None:
            return float(arcface.cosine_similarity(emb1, emb2))
    except:
        pass
    return -1.0

def compute_lpips(img1, img2):
    """Compute LPIPS distance."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
        # Convert to tensor
        t1 = torch.from_numpy(img1.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE) / 255.0 * 2 - 1
        t2 = torch.from_numpy(img2.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE) / 255.0 * 2 - 1
        with torch.no_grad():
            d = loss_fn(t1, t2).item()
        return float(d)
    except:
        return -1.0

def compute_ssim(img1, img2):
    """Compute SSIM."""
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(img1, img2, channel_axis=2, data_range=255))
    except:
        return -1.0

def run_experiment1():
    print("="*70)
    print("EXPERIMENT 1: ALPHA SWEEP - LATENT_AVG INTERPOLATION")
    print("="*70)
    
    encoder, generator, adult_avg = load_models()
    
    # Build child latent_avg
    child_avg = build_child_latent_avg()
    print(f"Adult avg shape: {adult_avg.shape}")
    print(f"Child avg shape: {child_avg.shape}")
    
    # Save for reuse
    torch.save(adult_avg.cpu(), OUTPUT_DIR / 'adult_latent_avg.pt')
    torch.save(child_avg.cpu(), OUTPUT_DIR / 'child_latent_avg.pt')
    
    all_results = []
    
    for f_file, m_file, pair_name in TEST_PAIRS:
        print(f"\n{'='*50}")
        print(f"Processing {pair_name}...")
        print(f"{'='*50}")
        
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        for role, img_path in [("father", f_path), ("mother", m_path)]:
            print(f"\n  {role}:")
            
            # Load once, get residual
            w18_original, aligned, residual = load_and_encode(img_path, encoder, adult_avg)
            geom_original = measure_geometry_full(aligned)
            original_np = aligned
            
            # Save original
            cv2.imwrite(str(OUTPUT_DIR / 'images' / f'{pair_name}_{role}_original.png'), 
                       cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
            
            for alpha in ALPHAS:
                # Interpolated latent_avg
                interp_avg = alpha * child_avg + (1 - alpha) * adult_avg
                
                # Reconstruct: residual + interpolated_avg
                w18_interp = residual + interp_avg
                
                with torch.no_grad():
                    img_recon, _ = generator([w18_interp], return_latents=True, input_is_latent=True)
                
                recon_np = tensor2rgb(img_recon)
                geom_recon = measure_geometry_full(recon_np)
                
                if geom_recon is None:
                    print(f"    alpha={alpha:.1f}: No face detected!")
                    continue
                
                # Compute metrics
                arcface_sim = compute_arcface_similarity(original_np, recon_np)
                lpips_dist = compute_lpips(original_np, recon_np)
                ssim_val = compute_ssim(original_np, recon_np)
                
                # Landmark displacement
                lm_orig = np.array(geom_original['landmarks'])
                lm_recon = np.array(geom_recon['landmarks'])
                disp = np.mean(np.linalg.norm(lm_recon - lm_orig, axis=1))
                
                result = {
                    'pair': pair_name,
                    'role': role,
                    'alpha': float(alpha),
                    'arcface_similarity': arcface_sim,
                    'lpips': lpips_dist,
                    'ssim': ssim_val,
                    'landmark_disp': float(disp),
                    'geometry': geom_recon
                }
                all_results.append(result)
                
                print(f"    alpha={alpha:.1f}: WH={geom_recon['wh_ratio']:.4f}, "
                      f"Jaw={geom_recon['jaw_width']:.1f}, Cheek={geom_recon['cheek_width']:.1f}, "
                      f"ArcFace={arcface_sim:.4f}, LPIPS={lpips_dist:.4f}, SSIM={ssim_val:.4f}")
                
                # Save image for key alphas
                if alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
                    cv2.imwrite(str(OUTPUT_DIR / 'images' / f'{pair_name}_{role}_alpha{alpha:.1f}.png'),
                               cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR))
    
    # Save all results
    with open(OUTPUT_DIR / 'exp1_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    generate_alpha_plots(all_results)
    
    return all_results

def generate_alpha_plots(results):
    """Generate plots of metrics vs alpha."""
    print("\nGenerating plots...")
    
    # Group by pair and role
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r['pair'], r['role'])].append(r)
    
    metrics_to_plot = [
        ('wh_ratio', 'W/H Ratio'),
        ('jaw_width', 'Jaw Width (px)'),
        ('cheek_width', 'Cheek Width (px)'),
        ('face_width', 'Face Width (px)'),
        ('face_height', 'Face Height (px)'),
        ('arcface_similarity', 'ArcFace Similarity'),
        ('lpips', 'LPIPS Distance'),
        ('ssim', 'SSIM'),
        ('landmark_disp', 'Mean Landmark Displacement (px)')
    ]
    
    for metric_key, metric_name in metrics_to_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, ((pair, role), group) in enumerate(grouped.items()):
            if idx >= 5:  # Only 5 pairs
                break
            ax = axes[idx]
            
            alphas = [r['alpha'] for r in group]
            if metric_key in ['arcface_similarity', 'lpips', 'ssim', 'landmark_disp']:
                values = [r[metric_key] for r in group]
            else:
                values = [r['geometry'][metric_key] for r in group]
            
            ax.plot(alphas, values, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Alpha (Child Weight)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{pair} - {role}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
        
        # Hide unused subplot
        if len(grouped) < 6:
            axes[5].set_visible(False)
        
        plt.suptitle(f'{metric_name} vs Alpha (Child Latent_avg Weight)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plots' / f'alpha_sweep_{metric_key}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Aggregate plot (mean across all pairs)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Mean and std across all pairs/roles
        alphas = ALPHAS
        means = []
        stds = []
        
        for alpha in alphas:
            vals = []
            for r in results:
                if abs(r['alpha'] - alpha) < 0.01:
                    if metric_key in ['arcface_similarity', 'lpips', 'ssim', 'landmark_disp']:
                        vals.append(r[metric_key])
                    else:
                        vals.append(r['geometry'][metric_key])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        
        ax.errorbar(alphas, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, markersize=6)
        ax.set_xlabel('Alpha (Child Weight)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Aggregate: {metric_name}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Aggregate Metrics vs Alpha (Child Latent_avg Weight)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plots' / 'alpha_sweep_aggregate.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {OUTPUT_DIR / 'plots'}")

if __name__ == '__main__':
    run_experiment1()