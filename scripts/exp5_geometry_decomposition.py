"""
Experiment 5: Geometry Decomposition
Mathematically decompose latent_final into latent_avg contribution + residual contribution.
Project each through StyleGAN and measure geometry contribution.
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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/e4e_geometric_bias_research/exp5_geometry_decomposition')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'images').mkdir(exist_ok=True)

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

def save_image(img_np, path):
    cv2.imwrite(str(path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

def run_experiment5():
    print("="*70)
    print("EXPERIMENT 5: GEOMETRY DECOMPOSITION - LATENT_AVG VS RESIDUAL")
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
            w18, aligned, residual = load_and_encode(img_path, encoder, latent_avg)
            
            # Component 1: latent_avg ONLY
            with torch.no_grad():
                img_latent_avg, _ = generator([latent_avg], return_latents=True, input_is_latent=True)
            img_latent_avg_np = tensor2rgb(img_latent_avg)
            geom_latent_avg = measure_geometry_full(img_latent_avg_np)
            
            # Component 2: residual ONLY (added to zero)
            w_residual_only = residual  # [1, 18, 512]
            with torch.no_grad():
                img_residual, _ = generator([w_residual_only], return_latents=True, input_is_latent=True)
            img_residual_np = tensor2rgb(img_residual)
            geom_residual = measure_geometry_full(img_residual_np)
            
            if geom_residual is None:
                print(f"    residual:     No face detected (invalid geometry)")
                # Use default values
                geom_residual = {
                    'wh_ratio': -1, 'jaw_width': -1, 'cheek_width': -1,
                    'face_width': -1, 'face_height': -1, 'temple_width': -1,
                    'chin_width': -1, 'forehead_width': -1,
                    'chin_nose': -1, 'nose_eye': -1, 'eye_forehead': -1,
                    'chin_forehead': -1, 'nose_mouth': -1, 'mouth_chin': -1,
                    'interocular_height': -1, 'face_bbox_height': -1,
                    'landmarks': []
                }
            else:
                print(f"    residual:     WH={geom_residual['wh_ratio']:.4f}, Jaw={geom_residual['jaw_width']:.1f}, Cheek={geom_residual['cheek_width']:.1f}")
            
            # Component 3: Full e4e (latent_avg + residual)
            with torch.no_grad():
                img_full, _ = generator([w18], return_latents=True, input_is_latent=True)
            img_full_np = tensor2rgb(img_full)
            geom_full = measure_geometry_full(img_full_np)
            
            # Original aligned image
            geom_original = measure_geometry_full(aligned)
            
            # Print summary
            print(f"    Original:     WH={geom_original['wh_ratio']:.4f}, Jaw={geom_original['jaw_width']:.1f}, Cheek={geom_original['cheek_width']:.1f}")
            print(f"    latent_avg:   WH={geom_latent_avg['wh_ratio']:.4f}, Jaw={geom_latent_avg['jaw_width']:.1f}, Cheek={geom_latent_avg['cheek_width']:.1f}")
            print(f"    residual:     WH={geom_residual['wh_ratio']:.4f}, Jaw={geom_residual['jaw_width']:.1f}, Cheek={geom_residual['cheek_width']:.1f}")
            print(f"    Full e4e:     WH={geom_full['wh_ratio']:.4f}, Jaw={geom_full['jaw_width']:.1f}, Cheek={geom_full['cheek_width']:.1f}")
            
            # Decomposition analysis
            # latent_avg provides baseline adult face
            # residual adds/subtracts from that baseline
            
            # Save images
            save_image(aligned, OUTPUT_DIR / 'images' / f'{pair_name}_{role}_original.png')
            save_image(img_latent_avg_np, OUTPUT_DIR / 'images' / f'{pair_name}_{role}_latent_avg.png')
            save_image(img_residual_np, OUTPUT_DIR / 'images' / f'{pair_name}_{role}_residual.png')
            save_image(img_full_np, OUTPUT_DIR / 'images' / f'{pair_name}_{role}_full_e4e.png')
            
            # Store results
            all_results.append({
                'pair': pair_name,
                'role': role,
                'original': geom_original,
                'latent_avg': geom_latent_avg,
                'residual': geom_residual,
                'full_e4e': geom_full
            })
    
    # Aggregate analysis
    print("\n" + "="*70)
    print("AGGREGATE DECOMPOSITION ANALYSIS")
    print("="*70)
    
    # Extract metrics
    metrics = ['wh_ratio', 'jaw_width', 'cheek_width', 'face_width', 'face_height', 
               'chin_nose', 'nose_eye', 'eye_forehead', 'chin_forehead']
    
    print("\n--- Component Means (10 faces) ---")
    for metric in metrics:
        orig = np.mean([r['original'][metric] for r in all_results])
        avg = np.mean([r['latent_avg'][metric] for r in all_results])
        res = np.mean([r['residual'][metric] for r in all_results])
        full = np.mean([r['full_e4e'][metric] for r in all_results])
        
        # Decomposition: full = f(latent_avg, residual)
        # In linear approximation: full ≈ latent_avg + residual (in latent space)
        # But generator is non-linear, so we measure actual geometry
        print(f"  {metric:15s}: Orig={orig:7.2f}, Avg={avg:7.2f}, Res={res:7.2f}, Full={full:7.2f}")
        
        # Contribution analysis
        if metric in ['wh_ratio', 'jaw_width', 'cheek_width', 'face_width', 'face_height']:
            avg_contrib = avg - orig  # How much latent_avg shifts from original
            res_contrib = full - orig  # Total change from original to full
            res_contrib_direct = res  # Residual geometry alone
            print(f"    -> latent_avg shifts {metric}: {avg_contrib:+.3f}")
            print(f"    -> residual geometry alone: {res_contrib_direct:+.3f}")
            print(f"    -> full e4e total change: {res_contrib:+.3f}")
    
    # Latent space norm analysis
    print("\n--- Latent Space Norm Analysis ---")
    for r in all_results:
        # We need to recompute norms
        pass
    
    # Compute latent norms for one example
    print("\n--- Latent Norms (P1 Father) ---")
    # We already have residual and latent_avg
    
    # Save all results
    with open(OUTPUT_DIR / 'exp5_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    return all_results

if __name__ == '__main__':
    run_experiment5()