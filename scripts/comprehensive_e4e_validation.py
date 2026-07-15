"""
Comprehensive e4e Geometric Bias Validation
Compares: Original Parent -> e4e W+ -> w_avg decoded
Measures geometry at each stage
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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/e4e_geometric_bias_research/comprehensive')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'images').mkdir(exist_ok=True)

def load_models():
    """Load encoder, generator, w_avg using proper key extraction"""
    ckp = torch.load('C:/tmp/ckpt/e4e_ffhq_encode.pt', map_location="cpu", weights_only=False)
    
    encoder = Encoder4Editing(50, "ir_se", 1024).eval().to(DEVICE)
    encoder.load_state_dict(get_keys(ckp, "encoder"), strict=True)
    
    w_avg = ckp["latent_avg"]  # [18, 512]
    w_avg_18 = w_avg.unsqueeze(0)  # [1, 18, 512]
    
    generator = Generator(1024, 512, 8).to(DEVICE)
    checkpoint = torch.load('C:/tmp/ckpt/stylegan2-ffhq-config-f.pt', map_location="cpu", weights_only=False)
    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    generator.eval()
    
    return encoder, generator, w_avg_18

def load_and_encode(image_path, encoder, w_avg_18):
    """Load, align, encode image to W+"""
    raw = cv2.imread(image_path)
    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    aligned = align_face(raw_rgb)
    
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    img_t = load_img(tmp.name).to(DEVICE)
    os.unlink(tmp.name)
    
    with torch.no_grad():
        w18 = encoder(torch.nn.functional.interpolate(img_t, size=(256, 256))) + w_avg_18.to(DEVICE)
    
    return w18, aligned

def tensor2rgb(tensor):
    tensor = (tensor * 0.5 + 0.5) * 255
    tensor = torch.clip(tensor, 0, 255).squeeze(0)
    return tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

def measure_geometry_full(img_np, pair_name, stage):
    """Measure comprehensive geometry"""
    landmarks = geom_estimator.get_landmarks(img_np)
    if landmarks is None:
        return None
    
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    # Core metrics
    face_width = dist(landmarks[0], landmarks[16])
    face_height = dist(landmarks[8], landmarks[27])
    wh_ratio = face_width / face_height if face_height > 0 else 0
    
    jaw_width = dist(landmarks[4], landmarks[12])
    cheek_width = dist(landmarks[2], landmarks[14])
    temple_width = dist(landmarks[1], landmarks[15])
    chin_width = dist(landmarks[6], landmarks[10])
    forehead_width = temple_width
    
    # Vertical segments
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
        'pair': pair_name,
        'stage': stage,
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
        'face_bbox_height': float(face_bbox_h)
    }

def save_image(img_np, path):
    cv2.imwrite(str(path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

def main():
    print("="*70)
    print("COMPREHENSIVE e4e GEOMETRIC BIAS VALIDATION")
    print("="*70)
    
    encoder, generator, w_avg_18 = load_models()
    
    # Decode w_avg once
    print("\n1. Decoding w_avg (latent_avg)...")
    with torch.no_grad():
        img_avg, _ = generator([w_avg_18.to(DEVICE)], return_latents=True, input_is_latent=True)
    img_avg_np = tensor2rgb(img_avg)
    save_image(img_avg_np, OUTPUT_DIR / 'images' / 'w_avg_decoded.png')
    geom_avg = measure_geometry_full(img_avg_np, 'w_avg', 'w_avg_decoded')
    print(f"   w_avg W/H: {geom_avg['wh_ratio']:.4f}, Jaw: {geom_avg['jaw_width']:.1f}, Cheek: {geom_avg['cheek_width']:.1f}")
    
    # Process each pair
    all_results = []
    
    for f_file, m_file, pair_name in TEST_PAIRS:
        print(f"\n{'='*50}")
        print(f"Processing {pair_name}...")
        print(f"{'='*50}")
        
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        for role, img_path in [("father", f_path), ("mother", m_path)]:
            # Stage 0: Original aligned
            w18, aligned = load_and_encode(img_path, encoder, w_avg_18)
            geom_orig = measure_geometry_full(aligned, pair_name, f'stage0_original_{role}')
            save_image(aligned, OUTPUT_DIR / 'images' / f'{pair_name}_{role}_stage0_original.png')
            print(f"   {role} Original: W={geom_orig['face_width']:.1f} H={geom_orig['face_height']:.1f} WH={geom_orig['wh_ratio']:.4f}")
            
            # Stage 1: e4e W+ reconstruction
            with torch.no_grad():
                img_recon, _ = generator([w18], return_latents=True, input_is_latent=True)
            img_recon_np = tensor2rgb(img_recon)
            geom_recon = measure_geometry_full(img_recon_np, pair_name, f'stage1_e4e_recon_{role}')
            save_image(img_recon_np, OUTPUT_DIR / 'images' / f'{pair_name}_{role}_stage1_e4e_recon.png')
            print(f"   {role} e4e Recon: W={geom_recon['face_width']:.1f} H={geom_recon['face_height']:.1f} WH={geom_recon['wh_ratio']:.4f}")
            
            # Deltas
            delta_wh = geom_recon['wh_ratio'] - geom_orig['wh_ratio']
            delta_w = geom_recon['face_width'] - geom_orig['face_width']
            delta_h = geom_recon['face_height'] - geom_orig['face_height']
            delta_jaw = geom_recon['jaw_width'] - geom_orig['jaw_width']
            delta_cheek = geom_recon['cheek_width'] - geom_orig['cheek_width']
            
            print(f"   DELTA: dWH={delta_wh:+.4f} dW={delta_w:+.1f} dH={delta_h:+.1f} dJaw={delta_jaw:+.1f} dCheek={delta_cheek:+.1f}")
            
            # Stage 2: w_avg (for reference)
            # Already measured above
            
            # Store results
            all_results.append({
                'pair': pair_name,
                'role': role,
                'original': geom_orig,
                'e4e_recon': geom_recon,
                'w_avg': geom_avg,
                'delta_e4e_vs_orig': {
                    'wh_ratio': delta_wh,
                    'face_width': delta_w,
                    'face_height': delta_h,
                    'jaw_width': delta_jaw,
                    'cheek_width': delta_cheek
                }
            })
    
    # Aggregate analysis
    print("\n" + "="*70)
    print("AGGREGATE ANALYSIS")
    print("="*70)
    
    # e4e vs original
    deltas = [r['delta_e4e_vs_orig'] for r in all_results]
    
    print("\n--- e4e Reconstruction vs Original (10 faces) ---")
    for metric in ['wh_ratio', 'face_width', 'face_height', 'jaw_width', 'cheek_width']:
        vals = [d[metric] for d in deltas]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        # t-test vs 0
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(vals, 0)
        d = mean_v / std_v if std_v > 0 else 0
        print(f"  {metric}: Δ={mean_v:+.4f} ± {std_v:.4f}, t={t_stat:.2f}, p={p_val:.4f}, d={d:.3f}")
    
    # w_avg vs typical
    print(f"\n--- w_avg (latent_avg) Geometry ---")
    print(f"  W/H Ratio: {geom_avg['wh_ratio']:.4f}")
    print(f"  Jaw Width: {geom_avg['jaw_width']:.1f}")
    print(f"  Cheek Width: {geom_avg['cheek_width']:.1f}")
    print(f"  Face Height: {geom_avg['face_height']:.1f}")
    
    # Compare to literature
    print(f"\n--- Comparison to Literature ---")
    print(f"  Adult W/H (literature):  1.25 - 1.35")
    print(f"  Child W/H (literature):  1.05 - 1.15")
    print(f"  w_avg W/H:               {geom_avg['wh_ratio']:.4f}")
    print(f"  Mean e4e Recon W/H:      {np.mean([r['e4e_recon']['wh_ratio'] for r in all_results]):.4f}")
    print(f"  Mean Original W/H:       {np.mean([r['original']['wh_ratio'] for r in all_results]):.4f}")
    
    # Save all
    output = {
        'w_avg_geometry': geom_avg,
        'per_face_results': all_results,
        'aggregate': {
            'e4e_delta_wh_mean': float(np.mean([d['wh_ratio'] for d in deltas])),
            'e4e_delta_wh_std': float(np.std([d['wh_ratio'] for d in deltas])),
            'e4e_delta_wh_t': float(stats.ttest_1samp([d['wh_ratio'] for d in deltas], 0)[0]),
            'e4e_delta_wh_p': float(stats.ttest_1samp([d['wh_ratio'] for d in deltas], 0)[1]),
        }
    }
    
    with open(OUTPUT_DIR / 'comprehensive_validation.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR / 'comprehensive_validation.json'}")
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    avg_delta = np.mean([d['wh_ratio'] for d in deltas])
    if avg_delta > 0.05:
        print(f"CONFIRMED: e4e adds +{avg_delta:.4f} W/H ratio (widening)")
        print("ROOT CAUSE: w_avg (latent_avg) has adult proportions (W/H={:.4f})".format(geom_avg['wh_ratio']))
        print("FIX: Interpolate w_avg with child latent_avg")
    else:
        print(f"No significant widening detected: +{avg_delta:.4f}")

if __name__ == '__main__':
    main()