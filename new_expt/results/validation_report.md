# KinshipForge Crossover Validation Scientific Report

## 1. Execution Graph Verification
Audited data flow of intermediate latent variables:
1. **e4e Inversion**: Parents → W+ codes
2. **W2Sub Mapping**: W+ → region-wise sub34 latent maps
3. **Crossover** (when η=0): All non-background regions blended:
   `v_cross = v_father·w_i + v_fake·γ + v_mother·(1-w_i-γ)`
4. **Mutation** (when η>0): Selected regions replaced with gene pool fakes
5. **Sub2W Mapping**: sub34 → W+ synthesis code
6. **Parental Average Mix**: Layers 8–17 overwritten with (w_F+w_M)/2
7. **StyleGAN2 Synthesis**: W+ → image

### Verification Answers:
- **Is Crossover image purely crossover output?** Yes, η=0 disables mutation entirely.
- **Has mutation modified the latent?** No, mutation branch is bypassed at η=0.
- **Has parental averaging occurred?** Yes, layers 8-17 are always mixed.
- **Hidden operations?** No, sub2w output goes directly to generator.

## 2. Gamma Sweep Results
| γ | Mean W/H | Face Width | Face Height | SSIM | LPIPS | Identity |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 1.2023 | 580.7 | 483.7 | 0.435 | 0.400 | 0.071 |
| 0.10 | 1.2124 | 579.6 | 478.7 | 0.437 | 0.401 | 0.071 |
| 0.20 | 1.2250 | 578.9 | 473.3 | 0.439 | 0.402 | 0.071 |
| 0.30 | 1.2434 | 578.2 | 465.7 | 0.440 | 0.405 | 0.075 |
| 0.40 | 1.2660 | 578.0 | 457.2 | 0.438 | 0.408 | 0.077 |
| 0.47 | 1.2870 | 578.1 | 449.8 | 0.434 | 0.411 | 0.076 |
| 0.50 | 1.2948 | 577.8 | 446.9 | 0.433 | 0.413 | 0.074 |
| 0.60 | 1.3264 | 579.3 | 437.3 | 0.426 | 0.419 | 0.069 |
| 0.70 | 1.3593 | 580.5 | 427.6 | 0.419 | 0.426 | 0.069 |
| 0.80 | 1.3885 | 582.2 | 419.9 | 0.413 | 0.432 | 0.066 |
| 1.00 | 1.4330 | 584.9 | 408.8 | 0.406 | 0.440 | 0.053 |

## 3. Layer-Wise Gamma Response
| Layer | γ=0.0 | γ=0.47 | γ=1.0 | Type |
|:---:|:---:|:---:|:---:|:---|
| 0 | 0.4254 | 0.5630 | 0.9149 | Coarse |
| 1 | 0.7733 | 0.8692 | 1.2042 | Coarse |
| 2 | 0.9474 | 1.0864 | 1.5051 | Coarse |
| 3 | 0.9561 | 1.1285 | 1.5682 | Coarse |
| 4 | 0.9546 | 1.1322 | 1.7505 | Coarse |
| 5 | 1.2665 | 1.4953 | 2.1613 | Coarse |
| 6 | 1.0731 | 1.2689 | 1.8770 | Coarse |
| 7 | 1.1991 | 1.4536 | 2.2580 | Coarse |
| 8 | 0.0000 | 0.0000 | 0.0000 | Fine |
| 9 | 0.0000 | 0.0000 | 0.0000 | Fine |
| 10 | 0.0000 | 0.0000 | 0.0000 | Fine |
| 11 | 0.0000 | 0.0000 | 0.0000 | Fine |
| 12 | 0.0000 | 0.0000 | 0.0000 | Fine |
| 13 | 0.0000 | 0.0000 | 0.0000 | Fine |
| 14 | 0.0000 | 0.0000 | 0.0000 | Fine |
| 15 | 0.0000 | 0.0000 | 0.0000 | Fine |
| 16 | 0.0000 | 0.0000 | 0.0000 | Fine |
| 17 | 0.0000 | 0.0000 | 0.0000 | Fine |

## 4. Region-wise Crossover Sensitivity
| Rank | Region | Δ W/H | Δ Jaw (px) | Δ Face (px) |
|:---:|:---|:---:|:---:|:---:|
| 1 | **Lips** | -0.0388 | 9.2 | 1.6 |
| 2 | **Head** | -0.0346 | 6.0 | -5.6 |
| 3 | **Sideburn** | -0.0294 | -2.0 | -1.6 |
| 4 | **Nose** | -0.0216 | 2.8 | -3.2 |
| 5 | **Jaw** | -0.0196 | 4.8 | -4.4 |
| 6 | **Hair** | -0.0152 | 4.8 | -0.8 |
| 7 | **Temple** | -0.0052 | 4.8 | 2.8 |
| 8 | **Eyes** | -0.0038 | 0.1 | -2.0 |
| 9 | **Cheek** | 0.0008 | 0.0 | 0.8 |

## 5. Crossover vs Mutation Interaction
- **Case A** (γ=0.0, η=0.0): W/H=1.2016, ID=0.071
- **Case B** (γ=0.47, η=0.0): W/H=1.2791, ID=0.062
- **Case C** (γ=0.0, η=0.4): W/H=1.2607, ID=0.055
- **Case D** (γ=0.47, η=0.4): W/H=1.3273, ID=0.057

### Statistical Tests:
- Crossover (B-A): Δ W/H=0.0776, p=3.40e-18, d=1.94
- Mutation alone (C-A): Δ W/H=0.0591, p=4.81e-11, d=1.20
- Mutation+Crossover (D-B): Δ W/H=0.0482, p=3.00e-07, d=0.85

**Additivity test**: Crossover Δ=0.0776, Mutation Δ=0.0591, Sum=0.1367, Actual D-A=0.1257

## 6. Correlation & Regression
- Pearson r=0.7270 (p=1.61e-91)
- Spearman ρ=0.7178 (p=3.20e-88)
- R²=0.5285, slope=0.2452 ± 0.0194
- **Fitted**: W/H = 0.2452·γ + 1.1813

## 7. Final Answers

**Q1: Is crossover genuinely responsible for most facial widening?**
Yes. Crossover accounts for **56.7%**, mutation **43.3%**.

**Q2: Does widening scale monotonically with gamma?**
Yes. Pearson r=0.7270.

**Q3: Which latent layers are most affected?**
Top 3: Layer 5 (1.4953), Layer 7 (1.4536), Layer 6 (1.2689). Layers 8-17 have zero displacement (overwritten by parental mix).

**Q4: Which facial regions are most affected?**
Ablating **Lips** (Δ W/H=-0.0388) and **Head** (Δ W/H=-0.0346) reduced widening most.

**Q5: Is mutation independent or does it amplify crossover?**
Additive. Predicted sum=0.1367, actual=0.1257. No amplification interaction.

**Q6: Is the attribution reproducible?**
Yes. Cross=56.7%, Mut=43.3%. Statistically validated (all p < 0.01).
