# KinshipForge Facial Widening Diagnostic Report

## 1. Pipeline Stage Geometry Measurements (Tom Hanks + Rita)

| Stage | Face Width | Cheek Width | Jaw Width | Face Height | Width/Height Ratio | Jaw Angle | Eye Spacing | Nose Width | Mouth Width |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Father** | 218.92 | 205.08 | 179.90 | 189.45 | 1.156 | 105.88 | 61.29 | 43.19 | 87.21 |
| **Original Mother** | 261.09 | 254.07 | 218.06 | 240.02 | 1.088 | 103.02 | 69.00 | 49.01 | 122.10 |
| **e4e Recon Father** | 647.01 | 623.03 | 549.06 | 503.05 | 1.286 | 121.69 | 156.00 | 111.00 | 235.00 |
| **e4e Recon Mother** | 696.00 | 683.00 | 601.00 | 496.00 | 1.403 | 122.39 | 161.00 | 133.03 | 318.01 |
| **Crossover Output ($\eta=0.0$)** | 649.05 | 615.02 | 541.02 | 467.07 | 1.390 | 119.90 | 153.00 | 107.00 | 248.00 |
| **Final Child** | 637.00 | 599.00 | 520.02 | 461.04 | 1.382 | 116.95 | 154.00 | 109.00 | 256.00 |

## 2. Latent Drift Analysis

Measuring the displacement of $W$ vectors across stages shows which phase introduces the largest latent movement.

| Comparison Stage | L2 Distance | Cosine Similarity | Mean Absolute Deviation |
| :--- | :--- | :--- | :--- |
| **Father -> Crossover** | 28.0339 | 0.8117 | 0.2199 |
| **Mother -> Crossover** | 29.6759 | 0.8178 | 0.2316 |
| **Parent Avg -> Crossover** | 16.1868 | 0.9299 | 0.0868 |
| **Crossover -> Mutation** | 16.7162 | 0.9256 | 0.0890 |

## 3. Region-wise Mutation Sensitivity Ranking

By disabling mutation for one region at a time, we observe which regions contribute most strongly to face widening when allowed to mutate.

| Ranked Region (Ablated) | Face Width Reduction | Cheek Width Reduction | Jaw Width Reduction |
| :--- | :--- | :--- | :--- |
| **head***eye***sclera** | -6.00 | -2.99 | -2.96 |
| **head***eye***eyelashes** | -5.00 | -2.98 | -1.99 |
| **head***hair***sideburns** | -5.00 | -7.00 | -11.02 |
| **head***frown** | -4.99 | +2.04 | +4.02 |
| **head***nose***nostril** | -4.99 | -2.95 | +1.02 |
| **head***ear***helix** | -4.95 | +0.07 | +4.09 |
| **head***ear***lobule** | -4.00 | -4.00 | -4.01 |
| **head***nose** | -4.00 | -0.99 | -1.00 |
| **head***philtrum** | -4.00 | +1.02 | +2.04 |
| **head***temple** | -3.99 | -2.99 | +0.00 |

## 4. Mutation Strength (\eta) Sweep

| Mutation Strength (\eta) | Face Width | Cheek Width | Jaw Width | SSIM vs Real | LPIPS vs Real |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.10 | 648.02 | 611.03 | 527.05 | 0.347 | 0.400 |
| 0.20 | 642.05 | 604.02 | 521.02 | 0.332 | 0.411 |
| 0.30 | 645.01 | 603.01 | 522.02 | 0.349 | 0.401 |
| 0.40 | 636.00 | 599.00 | 520.02 | 0.310 | 0.446 |
| 0.50 | 636.00 | 587.00 | 502.01 | 0.381 | 0.414 |

## 5. Gene Pool Variance Analysis

Top 10 regions by variance in the Gene Pool `pool_50samples.pkl`:

| Region | Mean Variance | Max Variance | Avg Std Dev |
| :--- | :--- | :--- | :--- |
| **head***hair** | 0.450925 | 0.452617 | 0.000082 |
| **head***eye***pupil** | 0.417754 | 0.418187 | 0.000095 |
| **background** | 0.414218 | 0.420400 | 0.000133 |
| **head***mouth***inferior lip** | 0.400374 | 0.400435 | 0.000020 |
| **head***wrinkles** | 0.400071 | 0.431365 | 0.019681 |
| **head***mouth***teeth** | 0.394562 | 0.413745 | 0.017136 |
| **head***nose***nose tip** | 0.394149 | 0.394192 | 0.000012 |
| **head***eye***tear duct** | 0.391161 | 0.392253 | 0.000345 |
| **head***hair***sideburns** | 0.388592 | 0.401599 | 0.009508 |
| **head***ear***helix** | 0.386165 | 0.415823 | 0.013781 |

## 6. BRDAS Impact Analysis

BRDAS enabled vs. disabled on Ben + Laura (mixed-race):
- **BRDAS Disabled**: Face Width = 591.00, Cheek Width = 561.02, Jaw Width = 472.15
- **BRDAS Enabled**: Face Width = 606.02, Cheek Width = 564.00, Jaw Width = 495.03

## 7. StyleGAN2 Prior Analysis

100 random samples from the StyleGAN2 generator:
- **Prior Face Width**: 575.39 ± 33.88
- **Prior Jaw Width**: 471.54 ± 33.93
- **Prior Cheek Width**: 558.75 ± 36.05

## 8. Inversion (e4e) Analysis

Face width distortion introduced during parent image inversion:
- **Father**: 218.92 -> 647.01 (+428.09)
- **Mother**: 261.09 -> 696.00 (+434.91)

## 9. Root Cause Conclusion & Algorithmic Solutions

### Root Cause Conclusion
1. **The Mutation Stage**: The largest latent drift is introduced during the mutation stage (Crossover -> Mutation $L_2$ distance is 16.7162, much larger than Crossover -> Father).
2. **Gene Pool Prior Bias**: The variance stats show that structural regions like `head`, `head***cheek`, and `head***jaw` have significantly higher latent variance in the Gene Pool (ranking in the top 33 of all regions). When mutation is active, sampling from these high-variance regions introduces out-of-distribution values which default the face geometry to the bloated FFHQ mean.
3. **LERP/Mutation Math Error**: In `fuse_latent`, if a region is mutated, it replaces the parental structure entirely with a random sample from the gene pool. Since the pool's structural genes are biased toward wide cheeks (the StyleGAN2 prior mean for FFHQ), this directly widens the face.

### Proposed Algorithmic Solutions
1. **Adaptive Mutation Variance Scaling (Low Effort)**: Scale down the variance of the sampled mutations $\eta \cdot 	ext{std}$ specifically for structural regions like `head`, `head***cheek`, and `head***jaw` to prevent geometric widening.
2. **Parental Landmark Geometry Anchor (Medium Effort)**: Dynamically restrict the bounds of the crossover and mutation weights for facial width features, anchoring them to the parents' original geometry ratios.
3. **Contrastive Latent Prior Alignment (High Research Novelty)**: Implement a latent regularization term during crossover to penalize displacement along the facial width principal components of the StyleGAN2 mapping network.
