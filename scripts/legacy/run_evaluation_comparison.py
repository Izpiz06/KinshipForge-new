import os
import sys
import gc
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import lpips
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn
from scipy import stats

# Resolve repository root
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

# Add StyleGene to path
sys.path.append(os.path.join(repo_root, 'StyleGene'))
sys.path.append(script_dir)

import configs
configs.path_ckpt_landmark68 = "C:/tmp/ckpt/shape_predictor_68_face_landmarks.dat.bz2"
configs.path_ckpt_e4e = "C:/tmp/ckpt/e4e_ffhq_encode.pt"
configs.path_ckpt_stylegan2 = "C:/tmp/ckpt/stylegan2-ffhq-config-f.pt"
configs.path_ckpt_stylegene = "C:/tmp/ckpt/stylegene_N18.ckpt"
configs.path_ckpt_fairface = "C:/tmp/ckpt/res34_fair_align_multi_7_20190809.pt"
configs.path_ckpt_genepool = "C:/tmp/ckpt/geneFactorPool.pkl"

import models.stylegene.api as api
from models.stylegene.data_util import face_class
from preprocess.align_images import align_face
from models.stylegene.util import load_img
from geometry_utils import GeometryEstimator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Init models
encoder, generator, sub2w, w2sub34, mean_latent = api.init_model()
encoder = encoder.to(device)
generator = generator.to(device)
sub2w = sub2w.to(device)
w2sub34 = w2sub34.to(device)
mean_latent = mean_latent.to(device)

loss_fn_lpips = lpips.LPIPS(net='alex').to(device).eval()
geom_estimator = GeometryEstimator()

# Define paths for test images
PAPA_PATH = os.path.join(repo_root, "pics/papa.jpeg")
MUMMA_PATH = os.path.join(repo_root, "pics/mumma.jpeg")
ME_PATH = os.path.join(repo_root, "pics/me.jpg")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tensor2rgb(tensor):
    t = (tensor * 0.5 + 0.5) * 255
    t = torch.clip(t, 0, 255).squeeze(0)
    return t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

def np_to_tensor(img_np):
    to_t = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return to_t(Image.fromarray(img_np).resize((256, 256))).unsqueeze(0).to(device)

def compute_geometry(img_np):
    landmarks = geom_estimator.get_landmarks(img_np)
    if landmarks is None:
        return None
    scale = 1024.0 / img_np.shape[0]
    landmarks = landmarks * scale
    def dist(p1, p2):
        return float(np.linalg.norm(p1 - p2))
    fw = dist(landmarks[0], landmarks[16])
    fh = dist(landmarks[8], landmarks[27])
    return {
        'Face Width': fw,
        'Face Height': fh,
        'Width/Height Ratio': fw / fh if fh > 0 else 0.0,
        'Jaw Width': dist(landmarks[4], landmarks[12]),
        'Cheek Width': dist(landmarks[2], landmarks[14]),
    }

try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _arcface_app = _FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    _arcface_app.prepare(ctx_id=0, det_size=(512, 512))
except:
    _arcface_app = None

def get_arcface_embedding(img_np):
    if _arcface_app is None:
        return None
    try:
        pil = Image.fromarray(img_np.astype(np.uint8)).resize((512, 512), Image.LANCZOS)
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        faces = _arcface_app.get(bgr)
        return faces[0].normed_embedding if len(faces) > 0 else None
    except:
        return None

# Load pool data
pool_path = os.path.join(repo_root, 'pkl/pool_50samples.pkl')
pool_data = torch.load(pool_path, map_location='cpu')
from models.stylegene.gene_pool import GenePoolFactory
geneFactor = GenePoolFactory(root_ffhq=None, device=device, mean_latent=mean_latent, max_sample=300)
geneFactor.pools = pool_data

def query_parent_pools(pool_age, gender, race_f, race_m):
    if race_f == race_m:
        entries = geneFactor(encoder, w2sub34, pool_age, gender, race_f)
        if not entries:
            for age in ['0-2', '3-9', '10-19', '20-29']:
                entries += geneFactor(encoder, w2sub34, age, gender, race_f)
        return entries
    fp = geneFactor(encoder, w2sub34, pool_age, gender, race_f)
    mp = geneFactor(encoder, w2sub34, pool_age, gender, race_m)
    return {"father_pool": fp, "mother_pool": mp}

# Load align parents & target child
raw_f = np.array(Image.open(PAPA_PATH).convert('RGB'))
raw_m = np.array(Image.open(MUMMA_PATH).convert('RGB'))
raw_me = np.array(Image.open(ME_PATH).convert('RGB'))
aligned_F = align_face(raw_f, output_size=1024)
aligned_M = align_face(raw_m, output_size=1024)
aligned_ME = align_face(raw_me, output_size=1024)

import tempfile
tmp_f = tempfile.NamedTemporaryFile(suffix='.png', delete=False); tmp_f.close()
tmp_m = tempfile.NamedTemporaryFile(suffix='.png', delete=False); tmp_m.close()
cv2.imwrite(tmp_f.name, cv2.cvtColor(aligned_F, cv2.COLOR_RGB2BGR))
cv2.imwrite(tmp_m.name, cv2.cvtColor(aligned_M, cv2.COLOR_RGB2BGR))
img_t_F = load_img(tmp_f.name).to(device)
img_t_M = load_img(tmp_m.name).to(device)
try: os.remove(tmp_f.name); os.remove(tmp_m.name)
except: pass

with torch.no_grad():
    w18_F = encoder(F.interpolate(img_t_F, size=(256, 256))) + mean_latent
    w18_M = encoder(F.interpolate(img_t_M, size=(256, 256))) + mean_latent

me_256 = np.array(Image.fromarray(aligned_ME).resize((256, 256)))
me_tensor = np_to_tensor(aligned_ME)
me_emb = get_arcface_embedding(aligned_ME)

pools = query_parent_pools('3-9', 'male', 'Indian', 'Indian')
from models.stylegene.api import brdas_sampler, BrdasList
from models.stylegene.gene_crossover_mutation import fuse_latent

# Configurations:
# 1. Original: gamma=0.47, lambda=0.0
# 2. Global Avg(ARCS): gamma=0.32724, lambda=0.0
# 3. Global 0.40: gamma=0.40, lambda=0.0
# 4. Global 0.35: gamma=0.35, lambda=0.0
# 5. Global 0.30: gamma=0.30, lambda=0.0
# 6. Global 0.25: gamma=0.25, lambda=0.0
# 7. Global 0.20: gamma=0.20, lambda=0.0
# 8. ARCS: gamma=0.47, lambda=1.0

CONFIGS = {
    'Original':   {'gamma': 0.47, 'arcs_lambda': 0.0},
    'Global_Avg': {'gamma': 0.32724, 'arcs_lambda': 0.0},
    'Global_40':  {'gamma': 0.40, 'arcs_lambda': 0.0},
    'Global_35':  {'gamma': 0.35, 'arcs_lambda': 0.0},
    'Global_30':  {'gamma': 0.30, 'arcs_lambda': 0.0},
    'Global_25':  {'gamma': 0.25, 'arcs_lambda': 0.0},
    'Global_20':  {'gamma': 0.20, 'arcs_lambda': 0.0},
    'ARCS':       {'gamma': 0.47, 'arcs_lambda': 1.0}
}

NUM_SEEDS = 50
results_data = {cfg_name: [] for cfg_name in CONFIGS}

print(f"Starting evaluation sweep over {NUM_SEEDS} seeds...")

for seed_idx in range(1, NUM_SEEDS + 1):
    current_seed = seed_idx * 100 + 42 # unique seeds
    
    # Store Original Image of this seed to compute MAE and pixel differences relative to it
    # We will resolve this on the first run of the configuration loop for this seed
    orig_img_np = None
    
    for cfg_name, cfg_params in CONFIGS.items():
        set_seed(current_seed)
        rf = brdas_sampler(pools["father_pool"], pools["mother_pool"]) if isinstance(pools, dict) else pools
        
        # Measure runtime and memory
        t_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        
        w_syn = fuse_latent(
            w2sub34, sub2w, w18_F, w18_M, rf, 
            fixed_gamma=cfg_params['gamma'], fixed_eta=0.4, 
            arcs_lambda=cfg_params['arcs_lambda']
        )
        
        with torch.no_grad():
            img_t, _ = generator([w_syn], input_is_latent=True, return_latents=True)
            img_t_512 = F.interpolate(img_t, size=(512, 512), mode='area')
            
        t_end = time.perf_counter()
        rtime = (t_end - t_start) * 1000.0 # ms
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        
        img_np = tensor2rgb(img_t_512)
        
        # If this is 'Original', save it as the base image for MAE calculations of this seed
        if cfg_name == 'Original':
            orig_img_np = img_np.copy()
            
        # Calculate geometry
        geom = compute_geometry(img_np)
        wh = geom['Width/Height Ratio'] if geom else 0.0
        fw = geom['Face Width'] if geom else 0.0
        fh = geom['Face Height'] if geom else 0.0
        jaw = geom['Jaw Width'] if geom else 0.0
        cheek = geom['Cheek Width'] if geom else 0.0
        
        # Calculate Identity
        emb = get_arcface_embedding(img_np)
        id_score = float(np.dot(emb, me_emb)) if emb is not None and me_emb is not None else 0.0
        
        # Calculate Image Quality vs Ground Truth
        img_256 = np.array(Image.fromarray(img_np).resize((256, 256)))
        ssim_val = float(ssim_fn(img_256, me_256, channel_axis=2, data_range=255))
        t_img = np_to_tensor(img_np)
        with torch.no_grad():
            lp_val = loss_fn_lpips(t_img, me_tensor).item()
            
        # Calculate Difference vs Original StyleGene
        if orig_img_np is not None:
            mae_val = float(np.mean(np.abs(img_np.astype(float) - orig_img_np.astype(float))))
            pixel_diff = float(np.sum(np.abs(img_np.astype(float) - orig_img_np.astype(float)) > 10.0) / img_np.size)
        else:
            mae_val = 0.0
            pixel_diff = 0.0
            
        results_data[cfg_name].append({
            'seed': current_seed,
            'wh_ratio': wh,
            'face_width': fw,
            'face_height': fh,
            'jaw_width': jaw,
            'cheek_width': cheek,
            'identity': id_score,
            'ssim': ssim_val,
            'lpips': lp_val,
            'mae': mae_val,
            'pixel_diff': pixel_diff,
            'runtime': rtime,
            'memory': mem_mb
        })
        
        del img_t, img_t_512, img_np, img_256, t_img, w_syn
        torch.cuda.empty_cache(); gc.collect()
        
    if seed_idx % 10 == 0:
        print(f"  Processed {seed_idx}/{NUM_SEEDS} seeds...")

# Compile statistical summaries
print("\n" + "="*80)
print("EVALUATION SWEEP RESULTS TABLE")
print("="*80)

# Print Header
print(f"{'Method':<20} | {'Avg Gamma':<10} | {'W/H':<8} | {'Identity':<8} | {'SSIM':<8} | {'LPIPS':<8} | {'MAE vs Orig':<11} | {'Runtime (ms)':<12}")
print("-"*108)

summary_stats = {}
for name, samples in results_data.items():
    whs = [s['wh_ratio'] for s in samples if s['wh_ratio'] > 0]
    ids = [s['identity'] for s in samples]
    ssims = [s['ssim'] for s in samples]
    lpipss = [s['lpips'] for s in samples]
    maes = [s['mae'] for s in samples]
    rtimes = [s['runtime'] for s in samples]
    
    # Calculate Mean & Standard Deviation
    m_wh, sd_wh = np.mean(whs), np.std(whs)
    m_id, sd_id = np.mean(ids), np.std(ids)
    m_ssim, sd_ssim = np.mean(ssims), np.std(ssims)
    m_lpips, sd_lpips = np.mean(lpipss), np.std(lpipss)
    m_mae = np.mean(maes)
    m_rtime = np.mean(rtimes)
    
    effective_gamma = CONFIGS[name]['gamma']
    if name == 'ARCS':
        effective_gamma = 0.32724
        
    print(f"{name:<20} | {effective_gamma:<10.5f} | {m_wh:<8.4f} | {m_id:<8.4f} | {m_ssim:<8.4f} | {m_lpips:<8.4f} | {m_mae:<11.4f} | {m_rtime:<12.1f}")
    
    summary_stats[name] = {
        'wh': (whs, m_wh, sd_wh),
        'identity': (ids, m_id, sd_id),
        'ssim': (ssims, m_ssim, sd_ssim),
        'lpips': (lpipss, m_lpips, sd_lpips)
    }

# Statistical Analysis: Paired t-tests and Cohen's d
# Group 1: Original vs ARCS
# Group 2: Global_Avg (crossover identical to ARCS) vs ARCS
# Group 3: Global_Avg vs Original
print("\n" + "="*80)
print("STATISTICAL COMPARISON TABLE (Paired t-tests & Cohen's d)")
print("="*80)

def compare_groups(g1_name, g2_name, metric_key):
    g1_samples = results_data[g1_name]
    g2_samples = results_data[g2_name]
    
    x = np.array([s[metric_key] for s in g1_samples])
    y = np.array([s[metric_key] for s in g2_samples])
    
    diff = x - y
    t_stat, p_val = stats.ttest_rel(x, y)
    
    # Cohen's d for paired samples
    cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    
    # 95% Confidence Interval for the difference
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci_margin = stats.t.ppf(0.975, df=len(diff)-1) * se
    mean_diff = np.mean(diff)
    
    return {
        'mean_diff': mean_diff,
        'ci': (mean_diff - ci_margin, mean_diff + ci_margin),
        't_stat': t_stat,
        'p_val': p_val,
        'cohen_d': cohen_d
    }

metrics_to_compare = {
    'wh_ratio': 'Width/Height Ratio',
    'identity': 'Identity Similarity',
    'ssim': 'SSIM Quality',
    'lpips': 'LPIPS Quality'
}

comparisons = [
    ('Original', 'ARCS', "Original vs ARCS"),
    ('Global_Avg', 'ARCS', "Global Reduced vs ARCS (Numerically Fair)"),
    ('Original', 'Global_Avg', "Original vs Global Reduced")
]

for g1, g2, label in comparisons:
    print(f"\n--- Comparison: {label} ({g1} -> {g2}) ---")
    print(f"{'Metric':<25} | {'Mean Diff':<10} | {'95% CI':<18} | {'t-stat':<8} | {'p-value':<10} | {'Cohen d':<8}")
    print("-"*86)
    for m_key, m_label in metrics_to_compare.items():
        c_res = compare_groups(g1, g2, m_key)
        print(f"{m_label:<25} | {c_res['mean_diff']:<10.4f} | [{c_res['ci'][0]:.4f}, {c_res['ci'][1]:.4f}] | {c_res['t_stat']:<8.3f} | {c_res['p_val']:<10.4e} | {c_res['cohen_d']:<8.3f}")

# Plot and save W/H comparison across configurations
# We'll save this plot to C:\Users\mdiza\.gemini\antigravity-ide\brain\4f759186-7e7b-43f0-859e-706918b7e530\
plot_cfgs = ['Original', 'Global_40', 'Global_35', 'Global_Avg', 'Global_30', 'Global_25', 'Global_20', 'ARCS']
plot_means = [np.mean([s['wh_ratio'] for s in results_data[cfg]]) for cfg in plot_cfgs]
plot_stds = [np.std([s['wh_ratio'] for s in results_data[cfg]]) for cfg in plot_cfgs]

plt.figure(figsize=(10, 5))
x_positions = range(len(plot_cfgs))
plt.bar(x_positions, plot_means, yerr=plot_stds, align='center', alpha=0.8, ecolor='black', capsize=6, color='#2980b9')
plt.xticks(x_positions, plot_cfgs, rotation=15)
plt.ylabel('Width/Height Ratio')
plt.title('Comparison of Width/Height Ratios Across Crossover Configurations (50 Seeds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(1.20, 1.40)
comparison_plot_path = os.path.join(repo_root, "new_expt/results/crossover_comparison_wh.png")
plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
plt.close()

# Copy plot to brain directory
import shutil
shutil.copy(comparison_plot_path, "C:/Users/mdiza/.gemini/antigravity-ide/brain/4f759186-7e7b-43f0-859e-706918b7e530/crossover_comparison_wh.png")

print(f"\nComparison bar plot saved to C:/Users/mdiza/.gemini/antigravity-ide/brain/4f759186-7e7b-43f0-859e-706918b7e530/crossover_comparison_wh.png")
