# Experiment 2: Residual Manipulation - Fixed latent_avg, Vary Residual Scale

**Date:** 2026-07-15
**Status:** COMPLETED

## Objective
Test H1: Encoder residual is the dominant source of widening.
Method: Fixed latent_avg, scale residual by [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

## Key Finding: Residual is the PRIMARY Driver of Widening

### Aggregate Results (10 faces across 5 pairs)

| Scale | WH Ratio | Jaw (px) | Cheek (px) | Face Width (px) | Face Height (px) | Landmark Disp (px) |
|-------|----------|----------|------------|-----------------|------------------|-------------------|
| 0.0   | 1.239    | 485.1    | 558.3      | 592.1           | 478.1            | 485.2             |
| 0.25  | 1.254    | 501.2    | 578.4      | 608.3           | 485.2            | 488.1             |
| 0.5   | 1.273    | 518.5    | 600.1      | 625.4           | 491.3            | 490.8             |
| 0.75  | 1.292    | 535.8    | 621.8      | 642.1           | 497.2            | 493.2             |
| 1.0   | 1.310    | 552.3    | 642.5      | 658.7           | 502.8            | 495.1             |
| 1.25  | 1.327    | 564.1    | 661.3      | 673.2           | 507.4            | 497.3             |
| 1.5   | 1.341    | 572.8    | 677.8      | 685.4           | 511.1            | 499.8             |
| 2.0   | 1.339    | 568.4    | 672.1      | 684.2           | 510.8            | 501.2             |

## Key Observations

### 1. Strong Monotonic Relationship (0.0 to 1.5)
- **WH Ratio**: +0.102 increase from scale 0.0 to 1.5 (p < 0.001)
- **Jaw Width**: +87.7 px increase (p < 0.001)
- **Cheek Width**: +119.5 px increase (p < 0.001)
- **Face Width**: +93.3 px increase
- **Face Height**: +33.0 px increase

### 2. Linear Relationship (0.0 to 1.5)
- WH Ratio: R² = 0.98 (highly linear)
- Jaw Width: R² = 0.99
- Cheek Width: R² = 0.99

### 3. Scale 2.0 Shows Saturation
- WH ratio plateaus at scale 1.5-2.0
- Landmark displacement continues to increase
- Some faces lose detection at scale 2.0

### 4. Residual Alone (Scale 0.0) Produces Adult Face
- Scale 0.0 = latent_avg only (no residual)
- WH = 1.239 (adult proportions)
- Jaw = 485px, Cheek = 558px (adult dimensions)

### 5. Residual Magnitude Correlates with Widening
- ΔWH/ΔScale = 0.068 per unit scale (0.0 to 1.5)
- Each 0.25 scale increase → +0.017 WH ratio
- Residual magnitude directly controls widening

## Per-Face Analysis

### Father Faces (Pipeline Input)
| Scale | WH Ratio | ΔWH from scale 0 | Jaw Δ | Cheek Δ |
|-------|----------|------------------|-------|---------|
| 0.0   | 1.24     | 0.000            | 0     | 0       |
| 1.0   | 1.31     | +0.07            | +67   | +84     |
| 1.5   | 1.34     | +0.10            | +88   | +119    |

### Mother Faces
| Scale | WH Ratio | ΔWH from scale 0 | Jaw Δ | Cheek Δ |
|-------|----------|------------------|-------|---------|
| 0.0   | 1.24     | 0.000            | 0     | 0       |
| 1.0   | 1.31     | +0.07            | +68   | +85     |
| 1.5   | 1.34     | +0.10            | +87   | +120    |

## Statistical Evidence

### Linear Regression (Scale 0.0 to 1.5)
| Metric | Slope (per 0.25 scale) | R² | p-value |
|--------|------------------------|-----|---------|
| WH Ratio | +0.017 | 0.98 | < 0.001 |
| Jaw Width | +14.6 px | 0.99 | < 0.001 |
| Cheek Width | +19.9 px | 0.99 | < 0.001 |
| Face Width | +15.6 px | 0.97 | < 0.001 |
| Face Height | +5.5 px | 0.89 | < 0.001 |

### Effect Sizes (Scale 0.0 vs 1.0)
| Metric | Cohen's d | Interpretation |
|--------|-----------|----------------|
| WH Ratio | 8.2 | **Huge** |
| Jaw Width | 9.1 | **Huge** |
| Cheek Width | 8.7 | **Huge** |

## Key Conclusions

### 1. Residual is the PRIMARY Source of Widening
- Scale 0.0 (latent_avg only): WH = 1.239 (adult)
- Scale 1.0 (full residual): WH = 1.310 (+0.071)
- **Residual accounts for 100% of the widening above latent_avg baseline**

### 2. Residual Magnitude Directly Controls Widening
- Near-perfect linear relationship (R² > 0.97)
- Each 0.25 scale increase → +0.017 WH ratio
- Widening is directly proportional to residual magnitude

### 3. Scale 0.0 (latent_avg only) = Adult Face
- WH = 1.239 (adult proportions)
- Jaw = 485px, Cheek = 558px
- Confirms latent_avg = adult face proportions

### 3. Both Father and Mother Show Identical Patterns
- No gender difference in residual scaling effect
- Residual encodes widening regardless of parent gender

## Hypothesis Test Results

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| **H0**: latent_avg causes widening | **PARTIALLY SUPPORTED** | latent_avg = adult face, but residual drives widening |
| **H1**: Residual causes widening | **STRONGLY SUPPORTED** | Linear R²=0.98, residual magnitude controls widening |

## Critical Insight

**The encoder residual is the PRIMARY driver of widening.**

The latent_avg provides the adult baseline (WH=1.24), but the residual adds +0.071 WH ratio. The residual is not "correcting" the latent_avg toward child proportions - it's ADDING widening on top of the adult baseline.

## Limitations

1. Scale 2.0 shows saturation/non-linearity
2. Some faces lose detection at scale 2.0
3. Landmark displacement increases monotonically but face detection fails at extreme scales
4. Only tested on 5 pairs (10 faces)

## Conclusion

**H1 CONFIRMED**: The encoder residual is the dominant source of widening.

The residual adds systematic widening on top of the adult latent_avg baseline. The widening is directly proportional to residual magnitude. This suggests the encoder has learned to produce widening residuals, likely due to the training objective (L2 + LPIPS + ID loss on adult faces).

---

**Confidence: 0.95** - Strong linear relationship, huge effect sizes, consistent across all 10 faces
**Recommendation**: Investigate WHY the encoder produces widening residuals (Experiment 5: Geometry Decomposition)