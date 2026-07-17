# KinshipForge — Age-Progressive Child Face Synthesis

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**Age-progressive child face synthesis from parental images**  
Built on StyleGene (CVPR 2023) · Kaggle T4 GPU

*Research Internship — IIT Bhilai MIST Lab*

---

## Overview

KinshipForge extends the StyleGene framework to generate **one consistent child face at three age stages** — 5-10, 11-15, and 16-21 years — from just two parent photographs. Unlike the original StyleGene paper which generates independent sibling-like children per pair, this project maintains identity consistency using a **frozen DNA seed** across all three age outputs.

**Key innovations** — ARCS (Adaptive Region-wise Crossover Scaling), BRDAS (Balanced Region-wise Dual-Ancestry Sampling) for mixed-race pairs, and Gender-Biased Layer Fusion.

---

## Pipeline

![Pipeline](pipeline.png)

---

## Contributions

| Contribution | Description |
|---|---|
| **Frozen DNA Seed** | Fixes crossover weights across all 3 age stages — same genetic blueprint, visible aging via pool variation |
| **LERP Bucket Blending** | Maps display ages (5-10, 11-15, 16-21) to nearest gene pool age buckets (3-9, 10-19, 20-29) |
| **Gender-biased Layer Fusion** | 70/30 father/mother weighting on StyleGAN2 geometry layers (8-11) for male child, inverted for female — texture layers (12-17) stay 50/50 |
| **ARCS** | Adaptive Region-sCal ed St — per-region mutation gamma tuned by measured geometric sensitivity (0.0008–0.0432) |
| **BRDAS** | Balanced Region-wise Dual-Ancestry Sampling — per-region coin-flip ancestry for mixed-race parents with full logging |
| **Multi-seed Selection** | Runs seeds [42, 123, 256], selects best via LPIPS age progression |
| **Gene Pool Rebuild** | Rebuilt 8.11 GB FFHQ gene pool — 56 demographic keys, up to 100 samples/bucket |

---

## Root Cause Analysis: e4e Encoder Geometric Bias

### The Problem
KinshipForge consistently produced "widened" child faces — faces appeared wider than parents. Through systematic falsification experiments, the root cause was traced to the e4e encoder residual, not the latent_avg vector as initially hypothesized.

### Root Cause

| Component | Contribution to Widening | Evidence |
|---|---|---|
| **Encoder Residual E(I)** | **~75%** (+0.113 WH ratio) | Exp 2: R²=0.98 linear; Exp 5: residual norm (252) > latent_avg norm (235) |
| **latent_avg (adult prior)** | ~25% (+0.037 WH ratio) | Exp 3: latent_avg alone = WH=1.22 (adult face) |
| **Generator non-linearity** | Amplifies residual | Residual alone invalid; with latent_avg → 3× widening |

### Mechanism
```
W+ = latent_avg + E(I)
     = adult_mean + encoder_residual
     = WH=1.22    + adds +0.11 WH ratio
     = WH=1.33 (widened)
```

**Residual norm (252) > latent_avg norm (235)** — the residual is the dominant vector.

Training loss `L = L2 + LPIPS + ID_loss + λ||E(I)||²` forces the residual to encode adult facial structure, with L2 regularization (λ=0.025) too weak to counteract.

---

## Results (Latest Evaluation)

Evaluated on **7 parent pairs across 5 ethnicities** including Indian pairs absent from all existing benchmark datasets.

| Metric | Mean | Threshold | Pairs above threshold |
|---|---|---|---|
| **SSIM vs real child** | **0.288** | 0.25 | 6/7 |
| **LPIPS age progression** | **0.234** | 0.20 | 6/7 |
| **ArcFace identity consistency** | **0.393** | 0.25 | 7/7 |

### Per-Pair Results

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

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Izpiz06/KinshipForge-new.git
cd KinshipForge-new
```

### 2. Clone StyleGene and install dependencies
```bash
git clone https://github.com/CVI-SZU/StyleGene.git
pip install -r requirements.txt
```

### 3. Download model checkpoints
From [HuggingFace: wmpscc/StyleGene_CKPT](https://huggingface.co/wmpscc/StyleGene_CKPT) to `/tmp/ckpt/`:
- `e4e_ffhq_encode.pt`
- `stylegan2-ffhq-config-f.pt`
- `stylegene_N18.ckpt`
- `res34_fair_align_multi_7_20190809.pt`
- `shape_predictor_68_face_landmarks.dat.bz2`

### 4. Download Gene Pool
Custom rebuilt gene pool (8.11 GB):  
[Kaggle: manaswimendhekar/stylegene-balanced-pool](https://www.kaggle.com/datasets/manaswimendhekar/stylegene-balanced-pool)

> **Note:** This Kaggle dataset is currently Private. Access can be granted to the evaluation committee upon request.

---

## Running the Pipeline

### Kaggle Notebook (Recommended)
Upload `kinshipforge-notebook.ipynb` to Kaggle with **T4 GPU** + **Internet** enabled.

| Dataset | Kaggle Path |
|---|---|
| Parent/child photos | `manaswimendhekar/locked-7-pairs/` |
| Gene Pool | `manaswimendhekar/stylegene-balanced-pool/pool_50samples.pkl` |

The notebook handles all setup: cloning StyleGene, downloading checkpoints, patching architecture, running inference, and evaluating results.

---

## Project Structure

```
KinshipForge-new/
├── kinshipforge-notebook.ipynb     # Main Kaggle notebook (32 cells, 20 code + 12 markdown)
├── pipeline.png                    # Architecture diagram
│
├── archive/                        # Ground-truth locked-7-pairs (real photos)
│   ├── father_p{1-7}.jpg
│   ├── mother_p{1-7}.jpg/.jpeg
│   └── child_p{1-7}.jpg/.png
│
├── scripts/                        # Pipeline & analysis scripts
│   ├── stage*_*.py                 # Pipeline stages 1-6
│   ├── exp*_*.py                   # Falsification experiments 1-5
│   ├── falsify_*.py                # Hypothesis falsification
│   ├── mix_ablation.py             # Gender-biased mixing ablation
│   ├── validate_mix_fix.py         # Mixing validation
│   └── legacy/                     # Archived diagnostic scripts
│
├── kinshipforge/                   # Core package
│   ├── metrics/
│   │   └── core.py                 # SSIM, LPIPS, ArcFace, Geometry metrics
│   └── experiments/                # Experiment logger (CSV history)
│
├── e4e_geometric_bias_research/    # Root cause analysis (5 experiments)
├── kinshipforge-research/          # Research documentation
├── pics/                           # Demo images
├── pkl/                            # Gene pool (download separately, gitignored)
└── StyleGene/                      # CVPR 2023 submodule (gitignored)
```

---

## Research Documentation

All falsification experiments and root cause analysis:

```
e4e_geometric_bias_research/
├── 01_latent_avg_validation.md      # Alpha sweep (methodologically blocked)
├── 02_residual_analysis.md          # Residual scaling (DECISIVE: 75% widening)
├── 03_noise_perturbation.md         # Latent manifold NOT biased
├── 05_geometry_decomposition.md     # Component decomposition (DECISIVE: 75%)
└── FINAL_ROOT_CAUSE_REPORT.md       # Complete analysis
```

Additional research reports in `kinshipforge-research/results/` cover StyleGene reverse engineering, BRDAS theory, ARCS theory, and evaluation methodology.

---

## Notebook Contents (32 cells)

| Cell | Section | Description |
|------|---------|-------------|
| 1-2 | Dependencies | Install packages, clone StyleGene repo |
| 3 | Checkpoints | Download 4 model weights + dlib landmarks |
| 4-5 | Config & Patching | Write config, patch gene_crossover_mutation.py (ARCS) and api.py (BRDAS) |
| 6-9 | Model Init | Load e4e, StyleGAN2, StyleGene mappers, 8.11 GB gene pool, FairFace |
| 10-11 | Input Data | List photos, encode test + reconstruction verification |
| 12 | Full Pipeline | Core inference: encode parents → loop 3 ages → crossover/ARCS/BRDAS → StyleGAN2 decode |
| 13 | Exploratory | Run all 7 pairs with seed=42 for Multi-seed Selection analysis |
| 14 | Final Generation | Locked best seeds per pair → definitive outputs saved to `./outputs_final/` |
| 15-16 | Evaluation | LPIPS, SSIM, ArcFace kinship, DeepFace age metrics |
| 19-20 | Pool Fortification | Encode new images and append to gene pool |

---

## Known Limitations

- **Age floor:** StyleGAN2 trained on FFHQ (adult faces) — 5-10 bucket appears ~12-14 years
- **Indian female pool critically sparse:** 0-2-female-Indian has only 1 sample (FFHQ Western bias)
- **FairFace unreliable on celebs:** Race labels hardcoded for all 7 evaluation pairs
- **No age estimator works** on synthetic child faces from FFHQ-trained models
- **8.11 GB gene pool** requires significant RAM (~16 GB free) to load

---

## References

1. Li et al., "StyleGene: Crossover and Mutation of Region-level Facial Genes for Kinship Face Synthesis," CVPR 2023
2. Richardson et al., "Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation," CVPR 2021
3. Karras et al., "Analyzing and Improving the Image Quality of StyleGAN," CVPR 2020
4. Kärkkäinen and Joo, "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age," WACV 2021
5. Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," CVPR 2019
6. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," CVPR 2018

---

## License

MIT — see `LICENSE`

---

## Contact

**Manaswi Mendhekar** — manaswimendhekar@gmail.com  
Research Intern, MIST Lab, IIT Bhilai · B.Tech CSE (AI), CSVTU Bhilai  

**Mohammad Izaan** — mdizaan1192@gmail.com  
Research Intern, MIST Lab, IIT Bhilai · B.Tech CSE (IoT), SRM Institute of Science and Technology
