# KinshipForge — Project Structure & Reference

---

## Directory Structure

```
KinshipForge-iz/
├── .gitignore
├── LICENSE
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── pipeline.png                          # Architecture diagram
├── kinshipforge-notebook.ipynb           # Main Kaggle notebook (inference + evaluation)
│
├── archive/                              # Ground-truth locked evaluation pairs (7 pairs)
│   ├── father_p{1-7}.jpg
│   ├── mother_p{1-7}.jpg/.jpeg
│   └── child_p{1-7}.jpg/.png
│
├── pics/                                 # Demo/reference images
│
├── pkl/                                  # Gene pool data (ignored by git)
│   ├── FairFace/                         # FairFace demographic splits
│   ├── utk/                              # UTKFace demographic splits
│   ├── pkdec.py                          # Pickle inspection utility
│   └── pool_50samples.pkl                # 8.7 GB rebuilt FFHQ gene pool (56 keys)
│
├── StyleGene/                            # CVPR 2023 StyleGene (locally patched, gitignored)
│   ├── configs.py                        # Patched with local checkpoint paths
│   ├── models/stylegene/
│   │   ├── api.py                        # Patched (BRDAS, gender-biased mix)
│   │   └── gene_crossover_mutation.py    # Patched (gender-biased mix, ARCS)
│   └── preprocess/align_images.py
│
├── kinshipforge/                         # Main Python package
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── core.py                       # SSIM, LPIPS, ArcFace, Geometry
│   └── experiments/
│       ├── experiment_logger.py
│       └── logger.py
│
├── scripts/                              # Runnable scripts
│   ├── stage1_e4e_inversion.py           # Pipeline stages (standalone)
│   ├── stage2_w2sub_decomposition.py
│   ├── stage3_regional_crossover.py
│   ├── stage4_mutation.py
│   ├── stage5_sub2w_reconstruction.py
│   ├── stage6_stylegan_synthesis.py
│   ├── exp1_alpha_sweep.py               # Falsification experiments
│   ├── exp2_residual_analysis.py
│   ├── exp3_noise_perturbation.py
│   ├── exp5_geometry_decomposition.py
│   ├── falsify_H1_w2sub.py              # Hypothesis falsification scripts
│   ├── falsify_H2_sub2w.py
│   ├── falsify_H3_jacobian.py
│   ├── falsify_geometry_h0_h1.py
│   ├── mix_ablation.py                  # Layer mixing ablation
│   ├── validate_mix_fix.py              # Mixing validation
│   ├── ablation_mix.py                  # Gender-biased mixing ablation
│   ├── comprehensive_e4e_geometry_test.py
│   ├── comprehensive_e4e_validation.py
│   ├── geometry_preservation.py
│   ├── layer_perturbation.py
│   ├── pipeline_geometry_tracking.py
│   ├── region_coupling.py
│   ├── representation_analysis.py
│   ├── w2sub_sub2w_invertibility.py
│   └── legacy/                           # Archived diagnostic scripts
│       ├── run_diagnostics.py
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
├── e4e_geometric_bias_research/         # Root cause analysis (5 experiments)
│   ├── 01_latent_avg_validation.md       # Alpha sweep (methodologically blocked)
│   ├── 02_residual_analysis.md           # Residual scaling (DECISIVE: 75% widening)
│   ├── 03_noise_perturbation.md          # Latent manifold NOT biased
│   ├── 05_geometry_decomposition.md      # Component decomposition (DECISIVE: 75%)
│   ├── FINAL_ROOT_CAUSE_REPORT.md        # Complete analysis
│   ├── fields.yaml / outline.yaml        # Research planning
│   └── exp{1,2,3,5}_*/                  # Experiment data (JSON results)
│
└── kinshipforge-research/               # Broader research documentation
    └── results/
        ├── 01_stylegene_reverse_engineering.md
        ├── 02_facial_widening_root_cause.md
        ├── 04_kinshipforge_contributions_review.md
        ├── 05_brdas_theoretical_review.md
        ├── 06_arcs_theoretical_review.md
        ├── 07_literature_review.md
        ├── 08_evaluation_metrics.md
        ├── 09_future_architecture.md
        ├── 10_honest_assessment.md
        └── final_report.md
```

---

## Key Components

### 1. **Notebook Pipeline** (`kinshipforge-notebook.ipynb`)
Complete Kaggle-ready inference pipeline (32 cells):
- **Cells 1-4:** Dependencies, StyleGene clone, checkpoint download, config injection
- **Cell 5:** Architecture patching — overwrites `api.py` & `gene_crossover_mutation.py` with KinshipForge contributions (ARCS, BRDAS, gender-biased mix)
- **Cells 6-9:** Model init, gene pool load, FairFace init, inversion test
- **Cells 10-11:** `full_pipeline()` — main inference with BRDAS + gender-biased mix
- **Cells 12-14:** Exploratory + Final generation (7 pairs, 3 ages each)
- **Cells 15-17:** Quantitative evaluation (SSIM, LPIPS, ArcFace, DeepFace)
- **Cells 18-19:** Gene pool fortification (batch extraction for dataset upgrade)

### 2. **Root Cause Analysis** (`e4e_geometric_bias_research/`)
5 falsification experiments proving the e4e encoder residual accounts for **~75%** of facial widening (vs ~25% from latent_avg).

### 3. **Evaluation Metrics** (`kinshipforge/metrics/core.py`)
| Metric | Purpose | Threshold |
|--------|---------|-----------|
| SSIM | Structural similarity to real child | ≥ 0.25 |
| LPIPS (5-10 vs 16-21) | Visible age progression | ≥ 0.20 |
| ArcFace Identity | Cross-age consistency | ≥ 0.25 |
| Geometry (MediaPipe) | Face width/height, jaw, cheek ratios | — |

---

## Latest Evaluation Results

| Pair | Mean SSIM | LPIPS Age Prog | ArcFace Identity |
|------|-----------|----------------|-----------------|
| p1 Shahrukh+Gauri (Indian×Indian) | **0.306** | 0.247 | 0.572 |
| p2 Jackie+Joan (E.Asian×E.Asian) | 0.258 | **0.307** | 0.326 |
| p3 Obama+Michelle (Black×Black) | 0.272 | 0.270 | 0.475 |
| p4 TomHanks+Rita (White×White) | **0.354** | 0.229 | 0.269 |
| p5 Ben+Laura (Black×White) | 0.234 | 0.215 | 0.431 |
| p6 Tiger+Elin (Black×White) | 0.267 | 0.220 | 0.325 |
| p7 Mark+Kelly (Latino×White) | 0.329 | 0.149 | 0.350 |
| **MEAN** | **0.288** | **0.234** | **0.393** |

**Thresholds:** SSIM≥0.25 (6/7), LPIPS≥0.20 (6/7), Identity≥0.25 (7/7)

---

## Required Checkpoints (download to `/tmp/ckpt/`)

| File | Source |
|------|--------|
| `e4e_ffhq_encode.pt` | HF: `wmpscc/StyleGene_CKPT` |
| `stylegan2-ffhq-config-f.pt` | HF: `wmpscc/StyleGene_CKPT` |
| `stylegene_N18.ckpt` | HF: `wmpscc/StyleGene_CKPT` |
| `res34_fair_align_multi_7_20190809.pt` | HF: `wmpscc/StyleGene_CKPT` |
| `shape_predictor_68_face_landmarks.dat.bz2` | HF: `wmpscc/StyleGene_CKPT` |

---

## Gene Pool

| File | Size | Keys | Description |
|------|------|------|-------------|
| `pkl/pool_50samples.pkl` | 8.7 GB | 56 | Rebuilt from FFHQ 70k, balanced by age×gender×race |
| `pkl/FairFace/` | — | — | Demographic splits |
| `pkl/utk/` | — | — | UTKFace splits |

**Bucket scheme:** `0-2`, `3-9`, `10-19`, `20-29` × `male/female` × 7 races
**Age mapping:** `5-10→3-9`, `11-15→10-19`, `16-21→20-29`

---

## Known Limitations

1. **Age floor:** StyleGAN2 trained on FFHQ (adult faces) → 5-10 bucket appears ~12-14 years
2. **Indian female pool:** Critically sparse (0-2-female-Indian: 1 sample)
3. **FairFace on celebs:** Unreliable → manual race labels hardcoded for 7 pairs
4. **No age estimator:** Works on synthetic child faces from FFHQ-trained models

---

## Quick Start

### Kaggle (Recommended)
1. Upload `kinshipforge-notebook.ipynb` to Kaggle
2. Enable **T4 GPU** + **Internet**
3. Add datasets: `manaswimendhekar/stylegene-balanced-pool`, `izpiz06/locked-7-pairs`
4. Run all cells sequentially

### Local
```bash
pip install -r requirements.txt
# Download checkpoints to /tmp/ckpt/
# Place gene pool at pkl/pool_50samples.pkl
python scripts/stage1_e4e_inversion.py
```

---

## License

MIT — see `LICENSE`

---

## Contact

**Manaswi Mendhekar** — manaswimendhekar@gmail.com — @MANASWI-MENDHEKAR
Research Intern, MIST Lab, IIT Bhilai
B.Tech CSE (AI), CSVTU Bhilai

**Mohammad Izaan** — mdizaan1192@gmail.com — @MohammadIzaan
Research Intern, MIST Lab, IIT Bhilai
B.Tech CSE (IoT), SRM Institute of Science and Technology
