# KinshipForge — Project Structure & Reference

**Last Updated:** 2026-07-13  
**Version:** 1.0 (post-cleanup)

---

## 📁 Directory Structure

```
KinshipForge-iz/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── test_pool.py                          # Quick gene pool validation
├── child_face_gradio_ui.py               # Gradio UI (Kaggle notebook compatible)
├── kinshipforge-notebook.ipynb           # Main Kaggle notebook (25MB, all cells)
├── grid_detached_child.png               # Results grid visualization
├── pipeline.png                          # Architecture diagram
│
├── archive/                              # Ground-truth locked evaluation pairs (7 pairs × 3 ages)
│   ├── father_p{1-7}.jpg
│   ├── mother_p{1-7}.jpg/.jpeg
│   └── child_p{1-7}.jpg/.png
│
├── pics/                                 # Demo/reference images
│   └── (various stock/reference photos)
│
├── pkl/                                  # Gene pool data
│   ├── FairFace/                         # FairFace demographic splits
│   ├── utk/                              # UTKFace demographic splits
│   ├── pkdec.py                          # Pickle inspection utility
│   └── pool_50samples.pkl                # 8.7 GB rebuilt FFHQ gene pool (56 keys, 100 samples/bucket)
│
├── StyleGene/                            # Git submodule (CVI-SZU/StyleGene CVPR 2023)
│   ├── configs.py                        # Patched with local checkpoint paths
│   ├── models/stylegene/                 # Core synthesis logic (patched at runtime)
│   └── preprocess/align_images.py        # dlib face alignment
│
├── kinshipforge/                         # Main Python package
│   ├── __init__.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── core.py                       # SSIM, LPIPS, ArcFace, Geometry, Performance
│   └── experiments/
│       ├── __init__.py
│       ├── experiment_logger.py          # CSV experiment tracking with git hash
│       ├── logger.py                     # Lightweight experiment logger
│       └── 2026-07-13_*/                 # Auto-generated experiment runs
│           ├── experiment_configuration.md
│           ├── quantitative_results.json
│           └── comparison_report.md
│
├── scripts/                              # Runnable experiment scripts
│   ├── ablation_mix.py                   # Layer mixing ablation (gender-biased)
│   ├── validate_mix_fix.py               # Validation of gender-biased mixing fix
│   └── legacy/                           # Archived diagnostic/validation scripts
│       ├── run_diagnostics.py            # 10-phase root-cause analysis
│       ├── run_complete_diagnostics.py
│       ├── run_complete_diagnostics_v2.py
│       ├── run_validation_sweep.py
│       ├── run_evaluation_comparison.py
│       ├── diagnose_arcs.py
│       ├── validate_arcs.py
│       ├── validate_brdas.py
│       ├── verify_brdas.py
│       ├── gene_pool.py
│       ├── geometry_utils.py
│       └── download_ckpts.py
│
├── kinshipforge-research/
│   └── results/                          # Research documentation (11 markdown files)
│       ├── 01_stylegene_reverse_engineering.md
│       ├── 02_facial_widening_root_cause.md
│       ├── 03_latent_geometry_analysis.md
│       ├── 04_kinshipforge_contributions_review.md
│       ├── 05_brdas_theoretical_review.md
│       ├── 06_arcs_theoretical_review.md
│       ├── 07_literature_review.md
│       ├── 08_evaluation_metrics.md
│       ├── 09_future_architecture.md
│       ├── 10_honest_assessment.md
│       └── final_report.md
│
├── docs/                                 # (Empty - reserved for future docs)
└── experiments/                          # Root-level experiment history
    ├── experiment_history.csv            # Master CSV log
    └── 2026-07-13_10-56-22_test_logger/  # Sample run
```

---

## 🔬 Key Components

### 1. **Notebook Pipeline** (`kinshipforge-notebook.ipynb`)
Complete Kaggle-ready inference pipeline:
- **Cells 1-3:** Dependencies, StyleGene clone, checkpoint download (5 files → `/tmp/ckpt/`)
- **Cell 4:** Config injection (Kaggle dataset paths)
- **Cell 5:** **Architecture patching** — overwrites `api.py` & `gene_crossover_mutation.py` with KinshipForge contributions:
  - Frozen DNA Seed (fixed crossover weights across age stages)
  - LERP Bucket Blending (intermediate age genes)
  - Gender-Biased Layer Fusion (70/30 at layers 8-11)
  - ARCS (Adaptive Region-wise Crossover Scaling)
  - BRDAS (Balanced Region-wise Dual-Ancestry Sampling)
- **Cells 6-9:** Model init, gene pool load (8.7GB), FairFace init
- **Cell 10:** e4e inversion sanity check
- **Cell 12:** `full_pipeline()` — main inference function
- **Cells 13-14:** Exploratory + Final generation (7 pairs, 3 ages each)
- **Cell 16:** Quantitative evaluation (SSIM, LPIPS age progression)
- **Cell 17:** Gradio UI launch

### 2. **Gradio UI** (`child_face_gradio_ui.py`)
- Standalone interface for manual testing
- Pre-cached mode (loads `outputs_final/`) + Live generation mode
- Auto seed selection (tries seeds 42, 123, 256 → picks max LPIPS age progression)
- Manual race selection (FairFace unreliable on celebrity photos)
- Metrics display: SSIM vs real child, LPIPS age progression, ArcFace identity consistency
- **Fixed:** Type-based filtering in save loops (skips `brdas_logs` metadata)

### 3. **Evaluation Metrics** (`kinshipforge/metrics/core.py`)
| Metric | Purpose | Threshold |
|--------|---------|-----------|
| SSIM | Structural similarity to real child | ≥ 0.25 |
| LPIPS (5-10 vs 16-21) | Visible age progression | ≥ 0.20 |
| ArcFace Identity | Cross-age consistency (5-10 vs 16-21) | ≥ 0.25 |
| Geometry (MediaPipe) | Face width/height, jaw, cheek ratios | — |

---

## 📊 Latest Evaluation Results (Notebook Cell 16)

| Pair | Mean SSIM | LPIPS Age Prog | Identity* |
|------|-----------|----------------|-----------|
| p1 Shahrukh+Gauri (Indian×Indian) | **0.306** | 0.247 | 0.572 |
| p2 Jackie+Joan (E.Asian×E.Asian) | 0.258 | **0.307** | 0.326 |
| p3 Obama+Michelle (Black×Black) | 0.272 | 0.270 | 0.475 |
| p4 TomHanks+Rita (White×White) | **0.354** | 0.229 | 0.269 |
| p5 Ben+Laura (Black×White) | 0.234 | 0.215 | 0.431 |
| p6 Tiger+Elin (Black×White) | 0.267 | 0.220 | 0.325 |
| p7 Mark+Kelly (Latino×White) | 0.329 | 0.149 | 0.350 |
| **MEAN** | **0.288** | **0.234** | **0.393** |

*Identity from Gradio UI cached metrics (ArcFace cosine similarity, 5-10 vs 16-21)

**Thresholds:** SSIM≥0.25 (6/7), LPIPS≥0.20 (6/7), Identity≥0.25 (7/7)

---

## 🧪 Experiment Scripts

| Script | Purpose | Run Location |
|--------|---------|--------------|
| `scripts/validate_mix_fix.py` | Layer mixing ablation (5 pairs × 4 variants × 10 seeds) | Local GPU / Kaggle |
| `scripts/ablation_mix.py` | Gender-biased mixing ablation with experiment logging | Local GPU / Kaggle |
| `scripts/legacy/run_diagnostics.py` | 10-phase root-cause analysis (geometry drift, pool variance, BRDAS, StyleGAN prior) | Local GPU |

---

## ⚙️ Required Checkpoints (download to `/tmp/ckpt/`)

| File | Source |
|------|--------|
| `e4e_ffhq_encode.pt` | HF: `wmpscc/StyleGene_CKPT` |
| `stylegan2-ffhq-config-f.pt` | HF: `wmpscc/StyleGene_CKPT` |
| `stylegene_N18.ckpt` | HF: `wmpscc/StyleGene_CKPT` |
| `res34_fair_align_multi_7_20190809.pt` | HF: `wmpscc/StyleGene_CKPT` |
| `shape_predictor_68_face_landmarks.dat.bz2` | HF: `wmpscc/StyleGene_CKPT` (auto-decompressed) |

---

## 🧬 Gene Pool

| File | Size | Keys | Description |
|------|------|------|-------------|
| `pkl/pool_50samples.pkl` | 8.7 GB | 56 | Rebuilt from FFHQ 70k, balanced by age×gender×race (100 samples/bucket where available) |
| `pkl/FairFace/` | — | — | FairFace demographic splits for pool reconstruction |
| `pkl/utk/` | — | — | UTKFace demographic splits for pool reconstruction |

**Bucket scheme:** `0-2`, `3-9`, `10-19`, `20-29` × `male/female` × 7 races  
**Age mapping:** `5-10→3-9`, `11-15→10-19`, `16-21→20-29`

---

## 📝 Research Documentation

Located in `kinshipforge-research/results/`:
- **01-03:** Root cause analysis (StyleGene reverse engineering, facial widening, latent geometry)
- **04:** KinshipForge contributions review
- **05-06:** BRDAS & ARCS theoretical reviews
- **07:** Literature review (kinship synthesis, age progression)
- **08:** Evaluation methodology
- **09:** Future architecture proposals
- **10:** Honest assessment (limitations, failure modes)
- **final_report.md:** Consolidated report

---

## 🚀 Quick Start

### Kaggle (Recommended)
1. Upload `kinshipforge-notebook.ipynb` to Kaggle
2. Enable **T4 GPU** + **Internet**
3. Add datasets:
   - `manaswimendhekar/stylegene-balanced-pool` (gene pool)
   - `izpiz06/locked-7-pairs` (evaluation photos)
   - `izpiz06/dataset-upgrade` (optional, for pool fortification)
4. Run all cells sequentially
5. Cell 17 launches Gradio UI at public URL

### Local (Windows/Linux)
```bash
# Prerequisites
pip install -r requirements.txt
# Download checkpoints to C:/tmp/ckpt/ (or /tmp/ckpt/)
# Place gene pool at pkl/pool_50samples.pkl

# Run validation
python scripts/validate_mix_fix.py

# Run ablation
python scripts/ablation_mix.py
```

---

## 🐛 Known Limitations

1. **Age floor:** StyleGAN2 trained on FFHQ (adult faces) → 5-10 bucket appears ~12-14 years
2. **Indian female pool:** Critically sparse (0-2-female-Indian: 1 sample)
3. **FairFace on celebs:** Unreliable → manual race labels hardcoded for 7 pairs
4. **Mixed-race BRDAS:** Only first parent's race used for mutation (bug, documented)
5. **No age estimator:** Works on synthetic child faces from FFHQ-trained models

---

## 📜 License

MIT — see `LICENSE`

---

## 📬 Contact

**Manaswi Mendhekar**  
Research Intern, MIST Lab, IIT Bhilai  
B.Tech CSE (AI), CSVTU Bhilai  
Email: manaswimendhekar@gmail.com  
GitHub: @MANASWI-MENDHEKAR