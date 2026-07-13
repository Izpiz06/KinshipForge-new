"""
KinshipForge — Child Face Age Progression UI
Built on StyleGene (CVPR 2023) with age progression extension.

REQUIRES (must be in notebook kernel before running this cell):
  - full_pipeline(path_father, path_mother, gender, child_seed, race_f, race_m) -> dict {"5-10": np.ndarray, "11-15": np.ndarray, "16-21": np.ndarray}
  - loss_fn_lpips  (lpips.LPIPS initialized in Cell 5)
  - device         (torch.device)

LAUNCH (in a Kaggle cell):
  import gradio as gr
  gr.close_all()
  with open('/kaggle/working/child_face_gradio_ui.py', 'r') as f:
      exec(f.read(), globals())
  demo = build_demo()
  demo.launch(share=True, server_name="0.0.0.0", server_port=7861, show_error=True)
"""

import gradio as gr
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import os
import traceback
import gc
import cv2
from insightface.app import FaceAnalysis as _FaceAnalysis
from skimage.metrics import structural_similarity as ssim_fn

# ─────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────
RACE_CHOICES = ["White", "Black", "Indian", "East Asian",
                "Southeast Asian", "Latino_Hispanic", "Middle Eastern"]

CANDIDATE_SEEDS = [42, 123, 256]

OUTPUTS_BASE = "/kaggle/input/datasets/YOUR_DATASET/outputs/outputs_final"
PHOTOS_BASE  = "/kaggle/input/datasets/YOUR_DATASET/locked-7-pairs"

# ─────────────────────────────────────────────────────────
# 2. PRE-CACHED PAIRS — updated confirmed metrics
# ─────────────────────────────────────────────────────────
CACHED_PAIRS = {
    "": {
        "father":  PHOTOS_BASE + "/father_p1.jpg",
        "mother":  PHOTOS_BASE + "/mother_p1.jpg",
        "child":   PHOTOS_BASE + "/child_p1.png",
        "outputs": {"5-10":  OUTPUTS_BASE + "/p1_shahrukh_gauri_5-10.png",
                    "11-15": OUTPUTS_BASE + "/p1_shahrukh_gauri_11-15.png",
                    "16-21": OUTPUTS_BASE + "/p1_shahrukh_gauri_16-21.png"},
        "metrics": {"ssim": 0.3056, "lpips_age": 0.2466, "identity": 0.572},
        "gender": "male", "race_f": "Indian", "race_m": "Indian",
    },
    "": {
        "father":  PHOTOS_BASE + "/father_p2.jpg",
        "mother":  PHOTOS_BASE + "/mother_p2.jpeg",
        "child":   PHOTOS_BASE + "/child_p2.jpg",
        "outputs": {"5-10":  OUTPUTS_BASE + "/p2_jackie_joan_5-10.png",
                    "11-15": OUTPUTS_BASE + "/p2_jackie_joan_11-15.png",
                    "16-21": OUTPUTS_BASE + "/p2_jackie_joan_16-21.png"},
        "metrics": {"ssim": 0.2578, "lpips_age": 0.3065, "identity": 0.326},
        "gender": "male", "race_f": "East Asian", "race_m": "East Asian",
    },
    "": {
        "father":  PHOTOS_BASE + "/father_p3.jpg",
        "mother":  PHOTOS_BASE + "/mother_p3.jpeg",
        "child":   PHOTOS_BASE + "/child_p3.jpg",
        "outputs": {"5-10":  OUTPUTS_BASE + "/p3_obama_michelle_5-10.png",
                    "11-15": OUTPUTS_BASE + "/p3_obama_michelle_11-15.png",
                    "16-21": OUTPUTS_BASE + "/p3_obama_michelle_16-21.png"},
        "metrics": {"ssim": 0.272, "lpips_age": 0.2699, "identity": 0.475},
        "gender": "female", "race_f": "Black", "race_m": "Black",
    },
    "": {
        "father":  PHOTOS_BASE + "/father_p4.jpg",
        "mother":  PHOTOS_BASE + "/mother_p4.jpg",
        "child":   PHOTOS_BASE + "/child_p4.jpg",
        "outputs": {"5-10":  OUTPUTS_BASE + "/p4_tomhanks_rita_5-10.png",
                    "11-15": OUTPUTS_BASE + "/p4_tomhanks_rita_11-15.png",
                    "16-21": OUTPUTS_BASE + "/p4_tomhanks_rita_16-21.png"},
        "metrics": {"ssim": 0.3535, "lpips_age": 0.2291, "identity": 0.269},
        "gender": "male", "race_f": "White", "race_m": "White",
    },
    "": {
        "father":  PHOTOS_BASE + "/father_p5.jpg",
        "mother":  PHOTOS_BASE + "/mother_p5.jpg",
        "child":   PHOTOS_BASE + "/child_p5.jpg",
        "outputs": {"5-10":  OUTPUTS_BASE + "/p5_ben_laura_5-10.png",
                    "11-15": OUTPUTS_BASE + "/p5_ben_laura_11-15.png",
                    "16-21": OUTPUTS_BASE + "/p5_ben_laura_16-21.png"},
        "metrics": {"ssim": 0.2342, "lpips_age": 0.2148, "identity": 0.431},
        "gender": "female", "race_f": "Black", "race_m": "White",
    },
    "": {
        "father":  PHOTOS_BASE + "/father_p6.jpg",
        "mother":  PHOTOS_BASE + "/mother_p6.jpg",
        "child":   PHOTOS_BASE + "/child_p6.jpg",
        "outputs": {"5-10":  OUTPUTS_BASE + "/p6_tiger_elin_5-10.png",
                    "11-15": OUTPUTS_BASE + "/p6_tiger_elin_11-15.png",
                    "16-21": OUTPUTS_BASE + "/p6_tiger_elin_16-21.png"},
        "metrics": {"ssim": 0.2668, "lpips_age": 0.2197, "identity": 0.325},
        "gender": "female", "race_f": "Black", "race_m": "White",
    },
    "": {
        "father":  PHOTOS_BASE + "/father_p7.jpg",
        "mother":  PHOTOS_BASE + "/mother_p7.jpg",
        "child":   PHOTOS_BASE + "/child_p7.jpg",
        "outputs": {"5-10":  OUTPUTS_BASE + "/p7_mark_kelly_5-10.png",
                    "11-15": OUTPUTS_BASE + "/p7_mark_kelly_11-15.png",
                    "16-21": OUTPUTS_BASE + "/p7_mark_kelly_16-21.png"},
        "metrics": {"ssim": 0.3286, "lpips_age": 0.1494, "identity": 0.35},
        "gender": "female", "race_f": "Latino_Hispanic", "race_m": "White",
    },
}


# ─────────────────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────────────────
def load_image_safe(path):
    if path and os.path.exists(path):
        return np.array(Image.open(path).convert("RGB").resize((256, 256)))
    return None


def np_to_tensor(img_np):
    to_t = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return to_t(Image.fromarray(img_np).resize((256, 256))).unsqueeze(0)


def compute_ssim(img1_np, img2_np):
    try:
        i1 = np.array(Image.fromarray(img1_np).resize((256, 256)))
        i2 = np.array(Image.fromarray(img2_np).resize((256, 256)))
        return round(float(ssim_fn(i1, i2, channel_axis=2, data_range=255)), 3)
    except Exception:
        return None


def compute_lpips_pair(img1_np, img2_np):
    try:
        t1 = np_to_tensor(img1_np).to(device)
        t2 = np_to_tensor(img2_np).to(device)
        with torch.no_grad():
            val = loss_fn_lpips(t1, t2).item()
        return round(val, 3)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# P20 FIX — ArcFace identity consistency via insightface
# ─────────────────────────────────────────────────────────
_arcface_app = None

def _get_arcface():
    global _arcface_app
    if _arcface_app is None:
        _arcface_app = _FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _arcface_app.prepare(ctx_id=0, det_size=(512, 512))
    return _arcface_app


def compute_identity_score(img1_np, img2_np):
    """
    ArcFace cosine similarity between two age stage outputs.
    Resizes to 1024x1024 before detection — insightface needs large images
    for synthetic GAN faces. Falls back to None if detection fails.
    """
    try:
        app = _get_arcface()

        def get_embedding(img_np):
            # resize to 1024 — critical for insightface on synthetic faces
            img_pil = Image.fromarray(img_np.astype(np.uint8)).resize((1024, 1024), Image.LANCZOS)
            bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            faces = app.get(bgr)
            print(f"  insightface detected {len(faces)} face(s), shape {bgr.shape}")
            if len(faces) == 0:
                return None
            return faces[0].normed_embedding  # already L2-normalised

        emb1 = get_embedding(img1_np)
        emb2 = get_embedding(img2_np)

        if emb1 is None or emb2 is None:
            return None

        # dot product of L2-normalised vectors = cosine similarity
        cos = float(np.dot(emb1, emb2))
        return round(cos, 3)

    except Exception as e:
        print(f"  identity score failed: {e}")
        return None


# ─────────────────────────────────────────────────────────
# 4. LIVE GENERATION
# ─────────────────────────────────────────────────────────
def run_live_generation(father_img, mother_img, gender, race_f, race_m, seed_choice):
    """
    Calls full_pipeline with user-specified races (no FairFace auto-detect).
    seed_choice: "Auto (best LPIPS)" tries all 3 seeds and picks best.
                 "42" / "123" / "256" / "512" / "1024" runs that seed only.
    """
    father_path = "/tmp/gradio_father.jpg"
    mother_path = "/tmp/gradio_mother.jpg"
    Image.fromarray(father_img).save(father_path)
    Image.fromarray(mother_img).save(mother_path)

    # Pre-flight checks
    missing = []
    for name, hint in [("full_pipeline", "run Cell 7 first"),
                       ("loss_fn_lpips", "run Cell 5 / LPIPS init first"),
                       ("device",        "run Cell 5a first")]:
        try:
            eval(name)
        except NameError:
            missing.append("{} ({})".format(name, hint))
    if missing:
        raise RuntimeError(
            "Missing kernel globals — run these cells first:\n  · " + "\n  · ".join(missing)
        )

    # Decide which seeds to try
    if seed_choice == "Auto (best LPIPS)":
        seeds_to_try = CANDIDATE_SEEDS
    else:
        seeds_to_try = [int(seed_choice)]

    best_results = None
    best_seed    = seeds_to_try[0]
    best_lpips   = -1.0
    seed_errors  = {}

    for seed in seeds_to_try:
        try:
            results = full_pipeline(
                path_father=father_path,
                path_mother=mother_path,
                gender=gender,
                child_seed=seed,
                race_f=race_f,
                race_m=race_m,
            )
            score = compute_lpips_pair(results["5-10"], results["16-21"])
            score = score if score is not None else 0.0
            if score > best_lpips:
                best_lpips   = score
                best_seed    = seed
                best_results = results
        except Exception as e:
            seed_errors[seed] = "{}: {}".format(type(e).__name__, str(e))
            continue

    if best_results is None:
        detail = "\n".join("  seed {}: {}".format(s, m) for s, m in seed_errors.items())
        raise RuntimeError(
            "All seeds failed in full_pipeline.\n\nPer-seed errors:\n{}\n\n"
            "Most likely causes:\n"
            "  1. Cell 7 not yet run — re-run cells 1-7 in order\n"
            "  2. geneFactor or fair_model not loaded — re-run cells 5c-5e\n"
            "  3. Checkpoint path wrong — verify Cell 4 shows /tmp/ckpt/\n"
            "  4. Kernel restarted — all cells must be re-run every session".format(detail)
        )

    gc.collect()
    torch.cuda.empty_cache()
    return best_results["5-10"], best_results["11-15"], best_results["16-21"], best_seed, round(best_lpips, 3), best_results.get("brdas_logs", {})


# ─────────────────────────────────────────────────────────
# 5. METRICS HTML
# ─────────────────────────────────────────────────────────
def _metric_card(label, value, hint, thresholds=None, baseline_txt=None):
    if value is None:
        val_html = "<span style='font-size:12px;color:#475569'>—</span>"
        interp   = ""
    else:
        val_str = "{:.3f}".format(value) if isinstance(value, float) else str(value)
        if thresholds is not None:
            strong, mod = thresholds
            if value >= strong:
                color       = "#16a34a"
                badge_bg    = "#14532d"
                badge_label = "▲ Strong"
            elif value >= mod:
                color       = "#d97706"
                badge_bg    = "#451a03"
                badge_label = "▲ Moderate"
            else:
                color       = "#dc2626"
                badge_bg    = "#450a0a"
                badge_label = "▼ Low"
            arrow  = "<span style='font-size:11px;font-weight:600;margin-left:8px;padding:2px 8px;border-radius:999px;background:{};color:{}'>{}</span>".format(
                badge_bg, color, badge_label)
            interp = "<div style='font-size:10px;color:{};margin-top:3px'>{}</div>".format(
                color, "Above threshold" if value >= strong else ("Near threshold" if value >= mod else "Below threshold"))
        else:
            color = "#94a3b8"; arrow = ""; interp = ""
        val_html = "<span style='font-size:24px;font-weight:600;color:{};letter-spacing:-0.5px'>{}</span>{}".format(
            color, val_str, arrow)
    baseline_row = ""
    if baseline_txt:
        baseline_row = "<div style='font-size:11px;color:#cbd5e1;margin-top:5px;border-top:1px solid #1e293b;padding-top:4px'>{}</div>".format(baseline_txt)
    return """<div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:14px 16px;'>
        <div style='font-size:12px;color:#e2e8f0;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.08em;font-weight:600'>{}</div>
        {}{}
        <div style='font-size:10px;color:#475569;margin-top:4px'>{}</div>
    </div>""".format(label, val_html, interp, baseline_row, hint)


def _build_metrics_html(ssim_val, lpips_age_val, identity_val, note, has_child, seed_used=None):
    seed_html = ""
    if seed_used is not None:
        seed_html = "<div style='font-size:10px;color:#334155;margin-bottom:8px'>Seed used: <span style='color:#3b82f6;font-family:DM Mono,monospace'>{}</span></div>".format(seed_used)
    ssim_card = _metric_card(
        "SSIM vs Real Child",
        ssim_val if has_child else None,
        "Higher = more similar to real child photo" if has_child else "Upload real child photo to compute",
        thresholds=(0.25, 0.18),
        baseline_txt="threshold 0.25 · structural similarity to ground truth" if has_child else None,
    )
    lpips_card = _metric_card(
        "LPIPS Age Progression", lpips_age_val,
        "Distance between age 5-10 and 16-21 outputs",
        thresholds=(0.20, 0.13),
        baseline_txt="threshold 0.20 · higher = more visible aging",
    )
    identity_card = _metric_card(
        "Identity Consistency", identity_val,
        "ArcFace cosine similarity: same child across ages?",
        thresholds=(0.25, 0.15),
        baseline_txt="threshold 0.25  · cross-age identity · validates frozen-seed contribution",
    )
    note_html = "<div style='font-size:11px;color:#475569;padding:6px 2px;border-top:1px solid #1e293b;margin-top:4px'>{}</div>".format(note)
    return """<div style='display:flex;flex-direction:column;gap:10px;'>
        {}
        <div style='font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-top:4px'>Evaluation metrics</div>
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;'>{}{}{}</div>
        {}
    </div>""".format(seed_html, ssim_card, lpips_card, identity_card, note_html)

# ─────────────────────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────────────────────
def step1_confirm(father_img, mother_img):
    if father_img is None or mother_img is None:
        return gr.update(visible=True), gr.update(interactive=False)
    return gr.update(visible=False), gr.update(interactive=True)


def on_confirm(f, m):
    return gr.update(interactive=True), f, m


def load_cached_pair(pair_name):
    """Only loads photos — does NOT auto-populate race or gender dropdowns."""
    if not pair_name or pair_name not in CACHED_PAIRS:
        return None, None, None
    pair = CACHED_PAIRS[pair_name]
    return (
        load_image_safe(pair["father"]),
        load_image_safe(pair["mother"]),
        load_image_safe(pair["child"]),
    )


def step2_generate(father_img, mother_img, child_img, gender,
                   race_f, race_m, seed_choice, mode, cached_pair_name):
    out_510 = out_1115 = out_1621 = None
    real_child_out = child_img
    ssim_val = lpips_age_val = identity_val = seed_used = None
    has_child = child_img is not None

    try:
        if mode == "Pre-cached (WiFi backup)":
            if not cached_pair_name or cached_pair_name not in CACHED_PAIRS:
                return None, None, None, None, "Select a pre-cached pair from the dropdown.", ""
            pair     = CACHED_PAIRS[cached_pair_name]
            out_510  = load_image_safe(pair["outputs"]["5-10"])
            out_1115 = load_image_safe(pair["outputs"]["11-15"])
            out_1621 = load_image_safe(pair["outputs"]["16-21"])
            if real_child_out is None:
                real_child_out = load_image_safe(pair["child"])
                has_child = real_child_out is not None
            m             = pair["metrics"]
            ssim_val      = m.get("ssim")      if has_child else None
            lpips_age_val = m.get("lpips_age")
            identity_val  = m.get("identity")
            note          = "Pre-cached results — " + cached_pair_name
            status_msg    = "Loaded: " + cached_pair_name

        else:
            # ── Live generation ──────────────────────────────
            if father_img is None or mother_img is None:
                return None, None, None, None, "Upload both parent photos first.", ""
            if not race_f or not race_m:
                return None, None, None, None, "Select father and mother race before generating.", ""

            out_510, out_1115, out_1621, seed_used, lpips_age_val, brdas_logs = run_live_generation(
                father_img, mother_img, gender, race_f, race_m, seed_choice
            )

            # P20 fix: identity consistency with fallback
            identity_val = compute_identity_score(out_510, out_1621)
            note_identity = ""
            if identity_val is None:
                # fallback: try 11-15 vs 16-21 if 5-10 detection failed
                identity_val = compute_identity_score(out_1115, out_1621)
                if identity_val is not None:
                    note_identity = " · identity: 11-15 vs 16-21 (5-10 det. failed)"
                else:
                    note_identity = " · identity: face detection failed"

            if has_child and out_510 is not None:
                ssim_val = compute_ssim(out_510, child_img)

            note       = "Live · seed {} · gender: {} · races: {} × {}{}".format(
                seed_used, gender, race_f, race_m, note_identity)
                
            # Format BRDAS log string for UI dashboard
            brdas_log_str = ""
            if brdas_logs:
                brdas_log_str = "\n\n--- BRDAS Demographic Region Logs ---"
                for age, selections in brdas_logs.items():
                    f_count = sum(1 for _, a in selections if a == "Father")
                    m_count = sum(1 for _, a in selections if a == "Mother")
                    brdas_log_str += f"\nAge {age} | Father regions: {f_count} | Mother regions: {m_count}"
                    for reg, anc in selections:
                        brdas_log_str += f"\n  - {reg}: {anc}"
            
            status_msg = "Generation complete. Seed: {}. LPIPS: {}. Races: {} × {}.{}".format(
                seed_used, lpips_age_val, race_f, race_m, brdas_log_str)
            if not has_child:
                note += " · upload real child photo to compute SSIM"

        metrics_html = _build_metrics_html(
            ssim_val, lpips_age_val, identity_val, note, has_child, seed_used)

    except Exception as e:
        status_msg   = "Error: {}\n\n{}".format(str(e), traceback.format_exc())
        metrics_html = _build_metrics_html(None, None, None, "Error — see status box.", False)

    return out_510, out_1115, out_1621, real_child_out, status_msg, metrics_html


# ─────────────────────────────────────────────────────────
# 7. STATIC CONTENT
# ─────────────────────────────────────────────────────────
LIMITATIONS_HTML = """
<div class='kf-lim-box'>
    <strong>Known limitations</strong>
    <div style='margin-top:8px;display:flex;flex-direction:column;gap:4px;'>
        <div class='kf-lim-row'><div class='kf-lim-dot'></div>
            <span>FFHQ age floor — StyleGAN2 trained on predominantly adult faces. The 5-10 bucket may appear approximately 12-14 years old. Hard generator limitation.</span></div>
        <div class='kf-lim-row'><div class='kf-lim-dot'></div>
            <span>Indian female pool sparsity — FFHQ underrepresents Indian female faces (0-2-female-Indian: 1 sample, 20-29-female-Indian: 19 samples).</span></div>
        <div class='kf-lim-row'><div class='kf-lim-dot'></div>
            <span>FairFace unreliable on celebrity photos — race selection is manual to avoid misclassification from makeup and studio lighting.</span></div>
        <div class='kf-lim-row'><div class='kf-lim-dot'></div>
            <span>Mixed-race pairs — Latino pool may be overridden by White pool when father is White. Known bug, documented as future work.</span></div>
        <div class='kf-lim-row'><div class='kf-lim-dot'></div>
            <span>Age accuracy not reported — no pretrained age estimator works on synthetic child faces from FFHQ-trained models. Shared limitation with the original StyleGene paper.</span></div>
        <div class='kf-lim-row'><div class='kf-lim-dot'></div>
            <span>16-21 bucket may appear older than 21 — StyleGAN2 latent space lacks a well-defined teen region due to adult-heavy FFHQ training data.</span></div>
    </div>
</div>
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
body, .gradio-container { font-family: 'DM Sans', sans-serif !important; background: #060c18 !important; }
.kf-header {
    background: linear-gradient(160deg, #0d1526 0%, #0a1020 60%, #0d1526 100%);
    padding: 32px 36px 26px; border-bottom: 1px solid #1a2540; margin-bottom: 0;
}
.kf-title { font-size: 28px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.8px; margin: 0 0 5px; }
.kf-subtitle { font-size: 12px; color: #475569; margin: 0 0 12px; letter-spacing: 0.02em; line-height: 1.6; }
.kf-stack-tag {
    display: inline-block; font-size: 10px; padding: 3px 9px; border-radius: 4px;
    background: #0f2340; color: #4a9eff; border: 1px solid #1a3a6b; margin-right: 5px;
    letter-spacing: 0.05em; text-transform: uppercase; font-family: 'DM Mono', monospace; font-weight: 500;
}
.kf-section {
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    color: #3b82f6; margin-bottom: 10px; margin-top: 4px; font-family: 'DM Mono', monospace;
}
.kf-lbl { font-size: 10px; text-align: center; color: #334155; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 5px; font-weight: 500; }
.kf-lbl-gen { font-size: 10px; text-align: center; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 5px; font-weight: 600; font-family: 'DM Mono', monospace; }
.kf-divider { border: none; border-top: 1px solid #1a2540; margin: 14px 0; }
.kf-lim-box { background: #0a1020; border: 1px solid #1a2540; border-radius: 10px; padding: 14px 18px; font-size: 11px; color: #475569; line-height: 2; }
.kf-lim-box strong { color: #64748b; font-weight: 600; }
.kf-lim-row { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 2px; }
.kf-lim-dot { width: 5px; height: 5px; border-radius: 50%; background: #334155; margin-top: 6px; flex-shrink: 0; }
.kf-status {
    font-family: 'DM Mono', monospace !important; font-size: 11px !important;
    color: #94a3b8 !important; background: #0a1020 !important;
    border: 1px solid #1a2540 !important; min-height: 120px !important;
}
"""


# ─────────────────────────────────────────────────────────
# 8. UI LAYOUT
# ─────────────────────────────────────────────────────────
def build_demo():
    with gr.Blocks(css=CUSTOM_CSS, title="KinshipForge") as demo:

        # ── Header ──────────────────────────────────────────
        gr.HTML("""
        <div class='kf-header'>
            <div class='kf-title'>KinshipForge</div>
            <div class='kf-subtitle'>
                Child face age progression from parental images<br>
                StyleGene backbone &nbsp;·&nbsp; LERP bucket blending &nbsp;·&nbsp;
                Gender-biased layer fusion &nbsp;·&nbsp; Multi-seed selection<br>
                Manaswi Mendhekar
            </div>
            <span class='kf-stack-tag'>StyleGAN2</span>
            <span class='kf-stack-tag'>e4e encoder</span>
            <span class='kf-stack-tag'>StyleGene</span>
            <span class='kf-stack-tag'>FairFace</span>
            <span class='kf-stack-tag'>7 pairs · 21 outputs</span>
        </div>
        """)

        # ── Mode + Pre-cached dropdown ───────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<div class='kf-section' style='margin-top:18px'>Mode</div>")
                mode = gr.Radio(
                    choices=["Live Generation", "Pre-cached (WiFi backup)"],
                    value="Live Generation", label="",
                )
            with gr.Column(scale=3):
                gr.HTML("<div class='kf-section' style='margin-top:18px'>Pre-cached pair</div>")
                gr.HTML("<div style='font-size:11px;color:#475569;margin-bottom:6px'>Select only in Pre-cached mode. In Live mode keep this blank — it does not affect generation.</div>")
                cached_pair = gr.Dropdown(
                    choices=[""] + list(CACHED_PAIRS.keys()),
                    value="",
                    label="",
                )

        gr.HTML("<hr class='kf-divider'>")

        # ── Step 1 — Upload ──────────────────────────────────
        gr.HTML("<div class='kf-section'>01 — Upload photos</div>")
        with gr.Row():
            with gr.Column():
                father_inp = gr.Image(label="Father", type="numpy", height=220)
                gr.HTML("<div class='kf-lbl'>Father</div>")
            with gr.Column():
                mother_inp = gr.Image(label="Mother", type="numpy", height=220)
                gr.HTML("<div class='kf-lbl'>Mother</div>")
            with gr.Column():
                child_inp  = gr.Image(label="Real child (optional)", type="numpy", height=220)
                gr.HTML("<div class='kf-lbl'>Real child &nbsp;·&nbsp; enables SSIM</div>")

        upload_warning = gr.HTML(
            "<p style='color:#f97316;font-size:12px;margin-top:6px'>Upload both parent photos to continue.</p>",
            visible=False,
        )

        # ── Step 2 — Gender + Race ───────────────────────────
        gr.HTML("<div class='kf-section' style='margin-top:14px'>02 — Select gender &amp; race</div>")
        gr.HTML("<div style='font-size:12px;color:#475569;margin-bottom:10px;line-height:1.7'>"
                "Race is selected manually — FairFace auto-detection is unreliable on celebrity photos "
                "(e.g. Shahrukh Khan detected as White, Gauri Khan as Black). "
                "Manual selection ensures correct gene pool query.</div>")
        with gr.Row():
            with gr.Column(scale=1):
                gender_sel = gr.Radio(choices=["male", "female"], value="male", label="Child gender")
            with gr.Column(scale=1):
                race_f_sel = gr.Dropdown(
                    choices=[""] + RACE_CHOICES, value="", label="Father race")
            with gr.Column(scale=1):
                race_m_sel = gr.Dropdown(
                    choices=[""] + RACE_CHOICES, value="", label="Mother race")
            with gr.Column(scale=1):
                confirm_btn = gr.Button("Confirm and proceed", interactive=False)

        gr.HTML("<hr class='kf-divider'>")

        # ── Step 3 — Seed + Generate ─────────────────────────
        gr.HTML("<div class='kf-section'>03 — Generate</div>")
        with gr.Row():
            with gr.Column(scale=3):
                seed_sel = gr.Dropdown(
                    choices=["Auto (best LPIPS)", "42", "123", "256", "512", "1024"],
                    value="Auto (best LPIPS)",
                    label="Seed / genetic variant",
                )
                gr.HTML("<div style='font-size:11px;color:#334155;margin-top:4px'>"
                        "Auto tries seeds [42, 123, 256] and picks highest LPIPS age-progression score. "
                        "Pick a fixed seed to get a specific genetic variant.</div>")
            with gr.Column(scale=2):
                generate_btn = gr.Button("Generate child faces", variant="primary", interactive=False)

        status_box = gr.Textbox(
            label="Status", value="Waiting for input...",
            interactive=False, lines=6, max_lines=20,
            elem_classes=["kf-status"],
        )

        gr.HTML("<hr class='kf-divider'>")

        # ── Results ──────────────────────────────────────────
        gr.HTML("<div class='kf-section'>Results</div>")
        with gr.Row():
            with gr.Column(min_width=130):
                out_father = gr.Image(label="Father",    interactive=False, height=200)
                gr.HTML("<div class='kf-lbl'>Father</div>")
            with gr.Column(min_width=130):
                out_mother = gr.Image(label="Mother",    interactive=False, height=200)
                gr.HTML("<div class='kf-lbl'>Mother</div>")
            with gr.Column(min_width=130):
                out_510  = gr.Image(label="Age 5–10",   interactive=False, height=200)
                gr.HTML("<div class='kf-lbl-gen'>Generated · 5–10</div>")
            with gr.Column(min_width=130):
                out_1115 = gr.Image(label="Age 11–15",  interactive=False, height=200)
                gr.HTML("<div class='kf-lbl-gen'>Generated · 11–15</div>")
            with gr.Column(min_width=130):
                out_1621 = gr.Image(label="Age 16–21",  interactive=False, height=200)
                gr.HTML("<div class='kf-lbl-gen'>Generated · 16–21</div>")
            with gr.Column(min_width=130):
                out_child = gr.Image(label="Real child", interactive=False, height=200)
                gr.HTML("<div class='kf-lbl'>Real child</div>")

        # ── Metrics ──────────────────────────────────────────
        gr.HTML("<div class='kf-section' style='margin-top:18px'>Evaluation metrics</div>")
        metrics_panel = gr.HTML(
            _build_metrics_html(None, None, None, "Run generation to see live metrics.", False)
        )

        # ── Limitations ──────────────────────────────────────
        gr.HTML("<div class='kf-section' style='margin-top:18px'>Known limitations</div>")
        gr.HTML(LIMITATIONS_HTML)
        gr.HTML("<div style='height:24px'></div>")

        # ── Callbacks ────────────────────────────────────────
        father_inp.change(fn=step1_confirm, inputs=[father_inp, mother_inp],
                          outputs=[upload_warning, confirm_btn])
        mother_inp.change(fn=step1_confirm, inputs=[father_inp, mother_inp],
                          outputs=[upload_warning, confirm_btn])
        confirm_btn.click(fn=on_confirm, inputs=[father_inp, mother_inp],
                          outputs=[generate_btn, out_father, out_mother])

        cached_pair.change(fn=load_cached_pair, inputs=[cached_pair],
                           outputs=[father_inp, mother_inp, child_inp])

        generate_btn.click(
            fn=step2_generate,
            inputs=[father_inp, mother_inp, child_inp, gender_sel,
                    race_f_sel, race_m_sel, seed_sel, mode, cached_pair],
            outputs=[out_510, out_1115, out_1621, out_child, status_box, metrics_panel],
        )

    return demo
