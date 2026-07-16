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
| **Frozen DNA Seed** | Fixes crossover weights (α, β) across all 3 age stages — same genetic blueprint, visible aging via pool variation |
| **LERP Bucket Blending** | Linearly interpolates FFHQ pool age buckets to create intermediate age genes for each output stage |
| **Gender-biased Layer Fusion** | 70/30 father/mother weighting on StyleGAN2 layers 8-17 for male, 30/70 for female — replaces paper's fixed 50/50 |
| **ARCS** | Adaptive Region-wise Crossover Scaling — per-region gamma tuned by measured geometric sensitivity (0.0008–0.0432) to minimize facial widening |
| **BRDAS** | Balanced Region-wise Dual-Ancestry Sampling — per-region coin-flip ancestry for mixed-race parents, with full logging |
| **Multi-seed Selection** | Runs seeds [42, 123, 256], selects seed with maximum LPIPS age progression — improved mean LPIPS from 0.207 to 0.267 |
| **Gene Pool Rebuild** | Rebuilt researchers' 27.8 GB inaccessible pool from FFHQ 70k — 56 keys, 100 samples/bucket, 8.71 GB |

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
git clone https://github.com/MANASWI-MENDHEKAR/KinshipForge.git
cd KinshipForge
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
Custom rebuilt gene pool (8.71 GB):  
[Kaggle: manaswimendhekar/stylegene-balanced-pool](https://www.kaggle.com/datasets/manaswimendhekar/stylegene-balanced-pool)

> **Note:** This Kaggle dataset is currently Private. Access can be granted to the evaluation committee upon request.

---

## Running the Pipeline

### Kaggle Notebook (Recommended)
Upload `kinshipforge-notebook.ipynb` to Kaggle with **T4 GPU** + **Internet** enabled.

| Dataset | Path |
|---|---|
| Parent/child photos | `YOUR_DATASET/locked-7-pairs/` |
| Gene Pool | `YOUR_DATASET/stylegene-balanced-pool/pool_50samples.pkl` |
| FFHQ 70k thumbnails | `YOUR_DATASET/ffhq-face-data-set/thumbnails128x128/` |

The notebook handles all setup: cloning StyleGene, downloading checkpoints, patching architecture, running inference, and evaluating results.

---

## Project Structure

```
KinshipForge/
├── kinshipforge-notebook.ipynb     # Main Kaggle notebook (32 cells)
├── requirements.txt                # Python dependencies
├── pipeline.png                    # Architecture diagram
│
├── kinshipforge/                   # Core package
│   ├── metrics/
│   │   └── core.py                 # SSIM, LPIPS, ArcFace, Geometry metrics
│   └── experiments/                # Experiment logger (CSV history)
│
├── scripts/                        # Standalone pipeline & analysis scripts
│   ├── stage*_*.py                 # Pipeline stages 1-6
│   ├── exp*_*.py                   # Falsification experiments 1-5
│   ├── falsify_*.py                # Hypothesis falsification
│   ├── mix_ablation.py             # Gender-biased mixing ablation
│   ├── validate_mix_fix.py         # Mixing validation
│   └── legacy/                     # Archived diagnostic scripts
│
├── archive/                        # Ground-truth locked-7-pairs (real photos)
│   ├── father_p{1-7}.jpg
│   ├── mother_p{1-7}.jpg/.jpeg
│   └── child_p{1-7}.jpg/.png
│
├── e4e_geometric_bias_research/    # Root cause analysis (5 experiments)
├── kinshipforge-research/          # Broader research documentation
├── pics/                           # Demo images
├── pkl/                            # Gene pool (gitignored, download separately)
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

## Known Limitations

- **Age floor:** StyleGAN2 trained on FFHQ (adult faces) — 5-10 bucket appears ~12-14 years
- **Indian female pool critically sparse:** 0-2-female-Indian has only 1 sample (FFHQ Western bias)
- **FairFace unreliable on celebs:** Race labels hardcoded for all 7 evaluation pairs
- **No age estimator works** on synthetic child faces from FFHQ-trained models

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
GitHub: [@MANASWI-MENDHEKAR](https://github.com/MANASWI-MENDHEKAR)

**Mohammad Izaan** — mdizaan1192@gmail.com  
Research Intern, MIST Lab, IIT Bhilai · B.Tech CSE (IoT), SRM Institute of Science and Technology  
GitHub: [@MohammadIzaan](https://github.com/MohammadIzaan)
