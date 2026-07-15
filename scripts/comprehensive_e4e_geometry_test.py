"""
Comprehensive test: Compare parent original vs e4e reconstruction vs w_avg decoded
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
from models.stylegene.util import load_img
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

OUTPUT_DIR = Path('C:/Users/mdiza/coding/KinshipForge-iz/e4e_geometric_bias_research/comprehensive')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_models():
    ckp = torch.load(path_ckpt_e4e, map_location="cpu", weights_only=False)
    encoder = Encoder4Editing(50, "ir_se", 1024).eval().to(DEVICE)
    encoder.load_state_dict(ckp["encoder"], strict=True)
    
    w_avg = ckp["latent_avg"]
    if w_avg.dim() == 1:
        w_avg_18 = w_avg.unsqueeze(0).repeat(1, 18, 1)
    else:
        w_avg_18 = w_avg.unsqueeze(0)
    
    generator = Generator(1024, 512, 8).to(DEVICE)
    checkpoint = torch.load(path_ckpt_stylegan2, map_location="cpu", weights_only=False)
    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    generator.eval()
    
    return encoder, generator, w_avg_18

def load_and_encode(image_path, encoder, w_avg_18):
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

def measure_all(landmarks):
    """Extended geometry measurement."""
    if landmarks is None:
        return None
    
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    # Face width (L0-L16)
    face_width = dist(landmarks[0], landmarks[16])
    
    # Face height (L8-L27)
    face_height = dist(landmarks[8], landmarks[27])
    
    # W/H ratio
    wh_ratio = face_width / face_height if face_height > 0 else 0
    
    # Jaw width (L4-L12)
    jaw_width = dist(landmarks[4], landmarks[12])
    
    # Cheek width (L2-L14)
    cheek_width = dist(landmarks[2], landmarks[14])
    
    # Temple width (L1-L15)
    temple_width = dist(landmarks[1], landmarks[15])
    
    # Chin width (L6-L10)
    chin_width = dist(landmarks[6], landmarks[10])
    
    # Forehead width (L1-L15 at brow)
    forehead_width = temple_width
    
    # Vertical segments
    # Chin to nose (L8-L30)
    chin_nose = dist(landmarks[8], landmarks[30])
    
    # Nose to eye (L30 - eye center)
    nose_tip = landmarks[30]
    left_eye = (landmarks[36] + landmarks[39]) / 2
    right_eye = (landmarks[42] + landmarks[45]) / 2
    eye_center = (left_eye + right_eye) / 2
    nose_eye = dist(nose_tip, eye_center)
    
    # Eye to forehead (eye center - L27)
    eye_forehead = dist(eye_center, landmarks[27])
    
    # Chin to forehead (L8-L27)
    chin_forehead = dist(landmarks[8], landmarks[27])
    
    # Nose to mouth (L30-L51)
    nose_mouth = dist(landmarks[30], landmarks[51])
    
    # Mouth to chin (L57-L8)
    mouth_chin = dist(landmarks[57], landmarks[8])
    
    # Interocular height
    interocular_height = abs(left_eye[1] - right_eye[1])
    
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
    }

def get_landmarks(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    rects = geom_estimator.detector(gray, 1)
    if len(rects) == 0:
        return None
    shape = geom_estimator.predictor(gray, rects[0])
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)

def save_landmarks_json(landmarks, path):
    if landmarks is not None:
        with open(path, 'w') as f:
            json.dump({'landmarks': landmarks.tolist()}, f, indent=2)

def main():
    encoder, generator, w_avg_18 = load_models()
    
    all_results = []
    
    for f_file, m_file, pair_name in TEST_PAIRS:
        print(f"\n{'='*60}")
        print(f"Processing {pair_name}...")
        print(f"{'='*60}")
        
        f_path = f"{PHOTOS}/{f_file}"
        m_path = f"{PHOTOS}/{m_file}"
        
        pair_results = {'pair': pair_name}
        
        for role, img_path in [("father", f_path), ("mother", m_path)]:
            print(f"\n  {role}:")
            
            # 1. Original aligned
            w18, aligned = load_and_encode(img_path, encoder, w_avg_18)
            lm_orig = get_landmarks(aligned)
            geom_orig = measure_all(lm_orig)
            
            # 2. e4e reconstruction
            with torch.no_grad():
                img_recon, _ = generator([w18], return_latents=True, input_is_latent=True)
            recon_np = tensor2rgb(img_recon)
            lm_recon = get_landmarks(recon_np)
            geom_recon = measure_all(lm_recon)
            
            # 3. Decode w_avg
            with torch.no_grad():
                img_avg, _ = generator([w_avg_18], return_latents=True, input_is_latent=True)
            avg_np = tensor2rgb(img_avg)
            lm_avg = get_landmarks(avg_np)
            geom_avg = measure_all(lm_avg)
            
            # Save landmarks
            save_landmarks_json(lm_orig, OUTPUT_DIR / f"{pair_name}_{role}_orig.json")
            save_landmarks_json(lm_recon, OUTPUT_DIR / f"{pair_name}_{role}_recon.json")
            save_landmarks_json(lm_avg, OUTPUT_DIR / f"{pair_name}_wavg.json")
            
            # Save images
            cv2.imwrite(str(OUTPUT_DIR / f"{pair_name}_{role}_orig.png"), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(OUTPUT_DIR / f"{pair_name}_{role}_recon.png"), cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(OUTPUT_DIR / f"{pair_name}_wavg.png"), cv2.cvtColor(avg_np, cv2.COLOR_RGB2BGR))
            
            # Compute deltas
            delta_e4e = {k: geom_recon[k] - geom_orig[k] for k in geom_orig}
            delta_wavg = {k: geom_avg[k] - geom_orig[k] for k in geom_orig}
            
            print(f"    Original:  W={geom_orig['face_width']:.1f}, H={geom_orig['face_height']:.1f}, WH={geom_orig['wh_ratio']:.4f}")
            print(f"    e4e Recon: W={geom_recon['face_width']:.1f}, H={geom_recon['face_height']:.1f}, WH={geom_recon['wh_ratio']:.4f}")
            print(f"    w_avg:     W={geom_avg['face_width']:.1f}, H={geom_avg['face_height']:.1f}, WH={geom_avg['wh_ratio']:.4f}")
            print(f"    Δ e4e:     ΔW={delta_e4e['face_width']:+.1f}, ΔH={delta_e4e['face_height']:+.1f}, ΔWH={delta_e4e['wh_ratio']:+.4f}")
            
            pair_results[role] = {
                'original': geom_orig,
                'reconstruction': geom_recon,
                'w_avg': geom_avg,
                'delta_e4e': delta_e4e,
                'delta_wavg': delta_wavg
            }
        
        all_results.append(pair_results)
    
    # Aggregate
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (Father only for pipeline tracking)")
    print(f"{'='*60}")
    
    for metric in ['face_width', 'face_height', 'wh_ratio', 'jaw_width', 'cheek_width', 'temple_width', 'chin_nose', 'nose_eye', 'eye_forehead', 'chin_forehead', 'nose_mouth', 'mouth_chin', 'interocular_height']:
        deltas_e4e = [r['father']['delta_e4e'][metric] for r in all_results if r['father']['delta_e4e'][metric] != 0]
        if deltas_e4e:
            mean_d = np.mean(deltas_e4e)
            std_d = np.std(deltas_e4e)
            print(f"  {metric}: Δ={mean_d:+.3f} ± {std_d:.3f} (n={len(deltas_e4e)})")
    
    # Save aggregate
    with open(OUTPUT_DIR / 'comprehensive_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == '__main__':
    main()