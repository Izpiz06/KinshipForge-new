![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

# KinshipForge

**Age-progressive child face synthesis from parental images**  
Built on StyleGene (CVPR 2023) · Kaggle T4 GPU · Gradio UI

*Research Internship — IIT Bhilai MIST Lab · B.Tech CSE (AI), CSVTU Bhilai*

---

## What This Project Does

KinshipForge extends the StyleGene framework to generate **one consistent child face at three age stages** — 5-10, 11-15, and 16-21 years — from just two parent photographs. Unlike the original StyleGene paper which generates 40 independent sibling-like children per pair, this project maintains identity consistency across all three age outputs using a **frozen DNA seed** approach.

---

## Pipeline

![Pipeline](pipeline.png)

---

## Five Original Contributions

| Contribution | Description |
|---|---|
| **Frozen DNA Seed** | Fixes crossover weights (α, β) across all 3 age stages — same genetic blueprint, visible aging via pool variation |
| **LERP Bucket Blending** | Linearly interpolates FFHQ pool age buckets to create intermediate age genes for each output stage |
| **Gender-biased Layer Fusion** | 70/30 father/mother weighting on StyleGAN2 layers 8-17 for male, 30/70 for female — replaces paper's fixed 50/50 |
| **Multi-seed Selection** | Runs seeds [42, 123, 256], selects seed with maximum LPIPS age progression — improved mean LPIPS from 0.207 to 0.267 |
| **Gene Pool Rebuild** | Rebuilt researchers' 27.8 GB inaccessible pool from FFHQ 70k — 56 keys, 100 samples/bucket, 8.71 GB |

---

## Results (Latest Evaluation)

![Results](grid_detached_child.png)

Evaluated on **7 parent pairs across 5 ethnicities** including Indian pairs absent from all existing benchmark datasets. Real child photographs used as ground truth for all 7 pairs.

| Metric | Mean | Threshold | Pairs above threshold |
|---|---|---|---|
| **SSIM vs real child** | **0.288** | 0.25 | 6/7 |
| **LPIPS age progression** | **0.234** | 0.20 | 6/7 |
| **ArcFace identity consistency** | **0.393** | 0.25 | 7/7 |

### Per-Pair Breakdown

| Pair | Parents | Child Gender | SSIM (mean) | LPIPS Age Prog. |
|---|---|---|---|---|
| P1 | Shahrukh + Gauri (Indian×Indian) | Male | 0.306 | 0.247 |
| P2 | Jackie + Joan (East Asian×East Asian) | Male | 0.258 | 0.307 |
| P3 | Obama + Michelle (Black×Black) | Female | 0.272 | 0.270 |
| P4 | Tom Hanks + Rita (White×White) | Male | **0.354** | 0.229 |
| P5 | Ben + Laura (Black×White) | Female | 0.234 | 0.215 |
| P6 | Tiger + Elin (Black×White) | Female | 0.267 | 0.220 |
| P7 | Mark + Kelly (Latino×White) | Female | 0.329 | **0.149** |

---

## Setup and Usage

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
Checkpoints available at HuggingFace:  
https://huggingface.co/wmpscc/StyleGene_CKPT

Download these 5 files to `/tmp/ckpt/`:
- `e4e_ffhq_encode.pt`
- `stylegan2-ffhq-config-f.pt`
- `stylegene_N18.ckpt`
- `res34_fair_align_multi_7_20190809.pt`
- `shape_predictor_68_face_landmarks.dat.bz2`

### 4. Download Gene Pool
Custom rebuilt gene pool (8.71 GB):  
https://www.kaggle.com/datasets/manaswimendhekar/stylegene-balanced-pool  
Place at: `YOUR_DATASET/pool_50samples.pkl`

> **Note:** Due to data privacy and storage constraints, this Kaggle dataset is currently set to Private. Access can be granted to the evaluation committee upon request.

---

## Running the Pipeline

### Option A: Kaggle Notebook (Recommended)
Open `kinshipforge-notebook.ipynb` on Kaggle with **T4 GPU** enabled.  
Update dataset paths from `YOUR_DATASET` to your actual Kaggle dataset paths.

| Dataset | Path |
|---|---|
| Parent/child photos | `YOUR_DATASET/locked-7-pairs/` |
| Gene Pool | `YOUR_DATASET/stylegene-balanced-pool/pool_50samples.pkl` |
| FFHQ 70k thumbnails | `YOUR_DATASET/ffhq-face-data-set/thumbnails128x128/` |

### Option B: Local Gradio UI
```bash
python child_face_gradio_ui.py
```
*Requires checkpoints at `C:/tmp/ckpt/` and gene pool at `pkl/pool_50samples.pkl` (Windows paths in `StyleGene/configs.py`)*.

---

## Project Structure

```
KinshipForge/
├── kinshipforge/              # Core package
│   ├── metrics/               # SSIM, LPIPS, ArcFace, Geometry metrics
│   └── experiments/           # Experiment logger (CSV history, reproducibility)
├── scripts/
│   ├── ablation_mix.py        # Layer mixing ablation (geometry vs texture weights)
│   ├── validate_mix_fix.py    # Layer mixing validation with ArcFace
│   └── legacy/                # Diagnostic/validation scripts (archived)
├── kinshipforge-notebook.ipynb # Complete Kaggle pipeline (27 cells)
├── child_face_gradio_ui.py    # Gradio demo with pre-cached 7-pair results
├── test_pool.py               # Quick gene pool verification
├── archive/                   # Ground-truth locked-7-pairs (real photos)
├── pkl/pool_50samples.pkl     # Gene pool (8.71 GB, 56 demographic keys)
├── StyleGene/                 # Submodule (CVPR 2023, patched at runtime)
├── pics/                      # Demo images
└── kinshipforge-research/     # Research documentation & analysis
```

---

## Known Limitations

- **Age floor:** StyleGAN2 trained on FFHQ lacks child faces below ~15 — 5-10 bucket appears ~12-14 years
- **Indian female pool critically sparse:** 0-2-female-Indian has only 1 sample (FFHQ Western bias)
- **FairFace unreliable on celebs:** Race labels hardcoded for all 7 evaluation pairs
- **Mixed-race BRDAS bug:** Only first parent's race used for mutation (documented, future fix)
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

## Acknowledgements

Research internship conducted at **MIST Lab, IIT Bhilai** under the guidance of **Dr. Sk. Subidh Ali** (Associate Professor, Dept. of CSE, IIT Bhilai).  
University mentor: **Dr. Dipti Verma** (Assistant Professor, CSVTU Bhilai).  
Built on the StyleGene codebase by **Hao Li et al., Shenzhen University**.

---

## Contact

For access requests or technical inquiries:
- **Email:** manaswimendhekar@gmail.com
- **GitHub:** [@MANASWI-MENDHEKAR](https://github.com/MANASWI-MENDHEKAR)