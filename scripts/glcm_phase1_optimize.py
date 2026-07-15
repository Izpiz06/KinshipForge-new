"""
Phase 1: Latent Optimization for Geometry Correction
Finds correction vectors ΔW that reduce widening while preserving identity.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import tempfile
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/e4e_geometric_bias_research/glcm_phase1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'images').mkdir(exist_ok=True)
(OUTPUT_DIR / 'latents').mkdir(exist_ok=True)
(OUTPUT_DIR / 'correction_vectors').mkdir(exist_ok=True)

OPTIMIZATION_CONFIG = {
    'n_iterations': 200,
    'lr': 0.02,
    'identity_weight': 0.50,
    'lpips_weight': 0.30,
    'geometry_weight': 0.20,
    'adam_betas': (0.9, 0.999),
    'adam_eps': 1e-8,
}

def load_models():
    ckp = torch.load(path_ckpt_e4e, map_location="cpu", weights_only=False)
    encoder = Encoder4Editing(50, "ir_se", 1024).eval().to(DEVICE)
    encoder.load_state_dict(get_keys(ckp, "encoder"), strict=True)
    
    latent_avg = ckp["latent_avg"].unsqueeze(0).to(DEVICE)
    
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
    
    return w18, aligned

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
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
        t1 = torch.from_numpy(img1.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE) / 255.0 * 2 - 1
        t2 = torch.from_numpy(img2.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE) / 255.0 * 2 - 1
        with torch.no_grad():
            d = loss_fn(t1, t2).item()
        return float(d)
    except:
        return -1.0

def compute_ssim(img1, img2):
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(img1, img2, channel_axis=2, data_range=255))
    except:
        return -1.0

def geometry_loss_fn(current_geom, target_geom):
    """Compute geometry loss between current and target geometry."""
    loss = 0.0
    # WH ratio error (scaled)
    loss += ((current_geom['wh_ratio'] - target_geom['wh_ratio']) / 0.1) ** 2
    # Jaw width error (normalized by ~100px)
    loss += ((current_geom['jaw_width'] - target_geom['jaw_width']) / 50.0) ** 2
    # Cheek width error (normalized by ~100px)
    loss += ((current_geom['cheek_width'] - target_geom['cheek_width']) / 50.0) ** 2
    # Temple width error
    loss += ((current_geom['temple_width'] - target_geom['temple_width']) / 50.0) ** 2
    # Face width error
    loss += ((current_geom['face_width'] - target_geom['face_width']) / 50.0) ** 2
    # Face height error (should not change much)
    loss += 0.5 * ((current_geom['face_height'] - target_geom['face_height']) / 50.0) ** 2
    return loss

def optimize_latent_for_geometry(w18_original, target_geom, original_img, generator, identity_weight, lpips_weight, geometry_weight, n_iterations, lr):
    """Optimize latent to match target geometry while preserving identity."""
    
    from kinshipforge.metrics import ArcFaceEvaluator
    import lpips
    
    w_optimized = w18_original.clone().detach().requires_grad_(True)
    
    # Identity embedding of original image
    with torch.no_grad():
        original_tensor = torch.from_numpy(original_img.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE) / 255.0
        arcface = ArcFaceEvaluator(DEVICE)
        original_emb = arcface.get_embedding(original_img)
        original_tensor_lpips = torch.from_numpy(original_img.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE) / 255.0 * 2 - 1
    
    # LPIPS loss
    lpips_loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
    
    optimizer = torch.optim.Adam([w_optimized], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    loss_history = []
    
    for i in range(n_iterations):
        optimizer.zero_grad()
        
        # Generate image
        img_recon, _ = generator([w_optimized], return_latents=True, input_is_latent=True)
        img_recon_np = tensor2rgb(img_recon)
        
        # Identity loss
        emb_recon = arcface.get_embedding(img_recon_np)
        if emb_recon is not None and original_emb is not None:
            identity_loss = 1.0 - arcface.cosine_similarity(original_emb, emb_recon)
        else:
            identity_loss = torch.tensor(0.0, device=DEVICE)
        
        # LPIPS loss - resize both to 256x256 for LPIPS
        img_recon_lpips = (img_recon * 0.5 + 0.5).clamp(0, 1)
        img_recon_lpips = torch.nn.functional.interpolate(img_recon_lpips, size=(256, 256), mode='bilinear', align_corners=False)
        img_recon_lpips = img_recon_lpips * 2 - 1
        
        original_tensor_lpips_256 = torch.nn.functional.interpolate(original_tensor_lpips, size=(256, 256), mode='bilinear', align_corners=False)
        
        lpips_loss = lpips_loss_fn(img_recon_lpips, original_tensor_lpips_256)
        
        # Geometry loss
        geom_recon = measure_geometry_full(img_recon_np)
        if geom_recon is not None:
            geom_loss = geometry_loss_fn(geom_recon, target_geom)
        else:
            geom_loss = torch.tensor(1000.0, device=DEVICE)
        
        # Total loss
        total_loss = (identity_weight * identity_loss + 
                     lpips_weight * lpips_loss + 
                     geometry_weight * geom_loss)
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append({
            'iter': i,
            'total': total_loss.item(),
            'identity': identity_loss.item() if isinstance(identity_loss, torch.Tensor) else identity_loss,
            'lpips': lpips_loss.item(),
            'geometry': geom_loss.item() if isinstance(geom_loss, torch.Tensor) else geom_loss
        })
        
        if i % 50 == 0:
            print(f"  Iter {i}: Total={total_loss.item():.4f}, ID={loss_history[-1]['identity']:.4f}, LPIPS={lpips_loss.item():.4f}, Geom={loss_history[-1]['geometry']:.4f}")
    
    return w_optimized.detach(), loss_history

def run_phase1():
    print("="*70)
    print("PHASE 1: LATENT OPTIMIZATION FOR GEOMETRY CORRECTION")
    print("="*70)
    
    encoder, generator, latent_avg = load_models()
    
    # Load LPIPS and ArcFace
    import lpips
    from kinshipforge.metrics import ArcFaceEvaluator
    
    all_correction_vectors = []
    all_metrics = []
    
    for f_file, m_file, pair_name in TEST_PAIRS:
        print(f"\n{'='*50}")
        print(f"Processing {pair_name}...")
        print(f"{'='*50}")
        
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        # Encode both parents
        w18_father, aligned_father = load_and_encode(f_path, encoder, latent_avg)
        w18_mother, aligned_mother = load_and_encode(m_path, encoder, latent_avg)
        
        # Measure original geometry
        geom_father = measure_geometry_full(aligned_father)
        geom_mother = measure_geometry_full(aligned_mother)
        
        # Define target geometry (average of parents)
        target_geom = {
            'wh_ratio': (geom_father['wh_ratio'] + geom_mother['wh_ratio']) / 2,
            'jaw_width': (geom_father['jaw_width'] + geom_mother['jaw_width']) / 2,
            'cheek_width': (geom_father['cheek_width'] + geom_mother['cheek_width']) / 2,
            'temple_width': (geom_father['temple_width'] + geom_mother['temple_width']) / 2,
            'face_width': (geom_father['face_width'] + geom_mother['face_width']) / 2,
            'face_height': (geom_father['face_height'] + geom_mother['face_height']) / 2,
        }
        
        print(f"  Father: WH={geom_father['wh_ratio']:.4f}, Jaw={geom_father['jaw_width']:.1f}, Cheek={geom_father['cheek_width']:.1f}")
        print(f"  Mother: WH={geom_mother['wh_ratio']:.4f}, Jaw={geom_mother['jaw_width']:.1f}, Cheek={geom_mother['cheek_width']:.1f}")
        print(f"  Target: WH={target_geom['wh_ratio']:.4f}, Jaw={target_geom['jaw_width']:.1f}, Cheek={target_geom['cheek_width']:.1f}")
        
        # Optimize father's latent
        print(f"\n  Optimizing father latent...")
        w18_father_opt, loss_hist_father = optimize_latent_for_geometry(
            w18_father, target_geom, aligned_father, generator,
            OPTIMIZATION_CONFIG['identity_weight'],
            OPTIMIZATION_CONFIG['lpips_weight'],
            OPTIMIZATION_CONFIG['geometry_weight'],
            OPTIMIZATION_CONFIG['n_iterations'],
            OPTIMIZATION_CONFIG['lr']
        )
        
        # Optimize mother's latent
        print(f"\n  Optimizing mother latent...")
        w18_mother_opt, loss_hist_mother = optimize_latent_for_geometry(
            w18_mother, target_geom, aligned_mother, generator,
            OPTIMIZATION_CONFIG['identity_weight'],
            OPTIMIZATION_CONFIG['lpips_weight'],
            OPTIMIZATION_CONFIG['geometry_weight'],
            OPTIMIZATION_CONFIG['n_iterations'],
            OPTIMIZATION_CONFIG['lr']
        )
        
        # Compute correction vectors
        delta_w_father = w18_father_opt - w18_father
        delta_w_mother = w18_mother_opt - w18_mother
        
        # Save latents and correction vectors
        torch.save(w18_father, OUTPUT_DIR / 'latents' / f'{pair_name}_father_original.pt')
        torch.save(w18_father_opt, OUTPUT_DIR / 'latents' / f'{pair_name}_father_optimized.pt')
        torch.save(delta_w_father, OUTPUT_DIR / 'correction_vectors' / f'{pair_name}_father_delta.pt')
        
        torch.save(w18_mother, OUTPUT_DIR / 'latents' / f'{pair_name}_mother_original.pt')
        torch.save(w18_mother_opt, OUTPUT_DIR / 'latents' / f'{pair_name}_mother_optimized.pt')
        torch.save(delta_w_mother, OUTPUT_DIR / 'correction_vectors' / f'{pair_name}_mother_delta.pt')
        
        # Evaluate results
        with torch.no_grad():
            img_father_orig = generator([w18_father], return_latents=True, input_is_latent=True)[0]
            img_father_opt = generator([w18_father_opt], return_latents=True, input_is_latent=True)[0]
            img_mother_orig = generator([w18_mother], return_latents=True, input_is_latent=True)[0]
            img_mother_opt = generator([w18_mother_opt], return_latents=True, input_is_latent=True)[0]
        
        img_father_orig_np = tensor2rgb(img_father_orig)
        img_father_opt_np = tensor2rgb(img_father_opt)
        img_mother_orig_np = tensor2rgb(img_mother_orig)
        img_mother_opt_np = tensor2rgb(img_mother_opt)
        
        # Measure geometry after optimization
        geom_father_orig = measure_geometry_full(tensor2rgb(img_father_orig))
        geom_father_opt = measure_geometry_full(tensor2rgb(img_father_opt))
        geom_mother_orig = measure_geometry_full(tensor2rgb(img_mother_orig))
        geom_mother_opt = measure_geometry_full(tensor2rgb(img_mother_opt))
        
        # Identity metrics
        arcface_father = compute_arcface_similarity(tensor2rgb(img_father_orig), tensor2rgb(img_father_opt))
        arcface_mother = compute_arcface_similarity(tensor2rgb(img_mother_orig), tensor2rgb(img_mother_opt))
        lpips_father = compute_lpips(tensor2rgb(img_father_orig), tensor2rgb(img_father_opt))
        lpips_mother = compute_lpips(tensor2rgb(img_mother_orig), tensor2rgb(img_mother_opt))
        ssim_father = compute_ssim(tensor2rgb(img_father_orig), tensor2rgb(img_father_opt))
        ssim_mother = compute_ssim(tensor2rgb(img_mother_orig), tensor2rgb(img_mother_opt))
        
        # Save images
        cv2.imwrite(str(OUTPUT_DIR / 'images' / f'{pair_name}_father_original.png'), cv2.cvtColor(tensor2rgb(img_father_orig), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUTPUT_DIR / 'images' / f'{pair_name}_father_optimized.png'), cv2.cvtColor(tensor2rgb(img_father_opt), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUTPUT_DIR / 'images' / f'{pair_name}_mother_original.png'), cv2.cvtColor(tensor2rgb(img_mother_orig), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUTPUT_DIR / 'images' / f'{pair_name}_mother_optimized.png'), cv2.cvtColor(tensor2rgb(img_mother_opt), cv2.COLOR_RGB2BGR))
        
        # Compute correction vector stats
        delta_w_father_np = delta_w_father.cpu().numpy()
        delta_w_mother_np = delta_w_mother.cpu().numpy()
        
        # Per-layer statistics
        layer_stats_father = []
        layer_stats_mother = []
        for layer in range(18):
            layer_delta_f = delta_w_father_np[0, layer, :]
            layer_delta_m = delta_w_mother_np[0, layer, :]
            layer_stats_father.append({
                'layer': layer,
                'l2_norm': float(np.linalg.norm(layer_delta_f)),
                'mean': float(np.mean(layer_delta_f)),
                'std': float(np.std(layer_delta_f))
            })
            layer_stats_mother.append({
                'layer': layer,
                'l2_norm': float(np.linalg.norm(layer_delta_m)),
                'mean': float(np.mean(layer_delta_m)),
                'std': float(np.std(layer_delta_m))
            })
        
        # Total correction vector stats
        total_delta_father = float(np.linalg.norm(delta_w_father_np))
        total_delta_mother = float(np.linalg.norm(delta_w_mother_np))
        
        # Cosine similarity between father and mother correction vectors
        cos_sim = np.dot(delta_w_father_np.flatten(), delta_w_mother_np.flatten()) / (
            np.linalg.norm(delta_w_father_np) * np.linalg.norm(delta_w_mother_np)
        )
        
        # Store results
        correction_data = {
            'pair': pair_name,
            'father': {
                'delta_w': delta_w_father.cpu().tolist(),
                'delta_norm': total_delta_father,
                'layer_stats': layer_stats_father,
                'geometry_before': geom_father,
                'geometry_after': measure_geometry_full(tensor2rgb(img_father_opt)),
                'identity_arcface': arcface_father,
                'lpips': lpips_father,
                'ssim': ssim_father,
            },
            'mother': {
                'delta_w': delta_w_mother.cpu().tolist(),
                'delta_norm': total_delta_mother,
                'layer_stats': layer_stats_mother,
                'geometry_before': geom_mother,
                'geometry_after': measure_geometry_full(tensor2rgb(img_mother_opt)),
                'identity_arcface': arcface_mother,
                'lpips': lpips_mother,
                'ssim': ssim_mother,
            },
            'target_geometry': target_geom,
            'cosine_similarity_father_mother': float(cos_sim),
        }
        
        all_correction_vectors.append(correction_data)
        
        # Print summary
        print(f"\n  Father: ΔWH={measure_geometry_full(tensor2rgb(img_father_opt))['wh_ratio'] - geom_father['wh_ratio']:+.4f}, "
              f"ΔJaw={measure_geometry_full(tensor2rgb(img_father_opt))['jaw_width'] - geom_father['jaw_width']:+.1f}, "
              f"ΔCheek={measure_geometry_full(tensor2rgb(img_father_opt))['cheek_width'] - geom_father['cheek_width']:+.1f}, "
              f"ArcFace={arcface_father:.4f}, LPIPS={lpips_father:.4f}")
        print(f"  Mother: ΔWH={measure_geometry_full(tensor2rgb(img_mother_opt))['wh_ratio'] - geom_mother['wh_ratio']:+.4f}, "
              f"ΔJaw={measure_geometry_full(tensor2rgb(img_mother_opt))['jaw_width'] - geom_mother['jaw_width']:+.1f}, "
              f"ΔCheek={measure_geometry_full(tensor2rgb(img_mother_opt))['cheek_width'] - geom_mother['cheek_width']:+.1f}, "
              f"ArcFace={arcface_mother:.4f}, LPIPS={lpips_mother:.4f}")
        print(f"  Cosine sim (father vs mother): {cos_sim:.4f}")
        
        # Save per-pair loss history
        with open(OUTPUT_DIR / 'correction_vectors' / f'{pair_name}_loss_history.json', 'w') as f:
            json.dump({
                'father': loss_hist_father,
                'mother': loss_hist_mother
            }, f, indent=2)
    
    # Save all correction vectors
    with open(OUTPUT_DIR / 'correction_vectors' / 'all_corrections.json', 'w') as f:
        json.dump(all_correction_vectors, f, indent=2)
    
    # Save all metrics for CSV export
    csv_rows = []
    for c in all_correction_vectors:
        for role in ['father', 'mother']:
            r = c[role]
            csv_rows.append({
                'pair': c['pair'],
                'role': role,
                'delta_norm': r['delta_norm'],
                'geometry_before_wh': r['geometry_before']['wh_ratio'],
                'geometry_before_jaw': r['geometry_before']['jaw_width'],
                'geometry_before_cheek': r['geometry_before']['cheek_width'],
                'geometry_after_wh': r['geometry_after']['wh_ratio'],
                'geometry_after_jaw': r['geometry_after']['jaw_width'],
                'geometry_after_cheek': r['geometry_after']['cheek_width'],
                'delta_wh': r['geometry_after']['wh_ratio'] - r['geometry_before']['wh_ratio'],
                'delta_jaw': r['geometry_after']['jaw_width'] - r['geometry_before']['jaw_width'],
                'delta_cheek': r['geometry_after']['cheek_width'] - r['geometry_before']['cheek_width'],
                'arcface': r['identity_arcface'],
                'lpips': r['lpips'],
                'ssim': r['ssim'],
            })
    
    import csv
    with open(OUTPUT_DIR / 'correction_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # Save all data
    with open(OUTPUT_DIR / 'phase1_results.json', 'w') as f:
        json.dump({
            'config': OPTIMIZATION_CONFIG,
            'timestamp': datetime.now().isoformat(),
            'corrections': all_correction_vectors
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print("PHASE 1 COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to {OUTPUT_DIR}")
    
    return all_correction_vectors

if __name__ == '__main__':
    run_phase1()