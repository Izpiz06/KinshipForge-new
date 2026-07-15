# Experiment 1: Alpha Sweep - Latent_avg Interpolation

**Date:** 2026-07-15
**Status:** COMPLETED - Methodological Issue Identified

## Objective
Test H0: Widening is caused by latent_avg bias by interpolating between adult (FFHQ) and child latent averages.

## Methodology
- Interpolation: `latent = residual + alpha * child_avg + (1-alpha) * adult_avg`
- Alpha sweep: 0.0, 0.1, ..., 1.0
- 5 parent pairs × 2 roles × 11 alphas = 110 reconstructions
- Metrics: WH ratio, jaw width, cheek width, ArcFace identity, LPIPS, SSIM

## Critical Finding: Synthetic Child_avg is Invalid

The child latent_avg was constructed by averaging 5000 random latents from StyleGAN's mapping network. This does **not** produce a child face latent - it produces random noise around the adult manifold.

| Metric | Result |
|--------|--------|
| ArcFace (alpha=0.0) | 0.06 - 0.56 (very low) |
| ArcFace (alpha=1.0) | 0.08 - 0.56 (very low) |
| WH ratio trend | No clear monotonic trend |
| Identity preservation | FAILED (< 0.6 for all) |

## Alpha Sweep Results (Aggregate)

### Father Faces
| Alpha | WH Ratio | Jaw (px) | Cheek (px) | ArcFace |
|-------|----------|----------|------------|---------|
| 0.0 | 1.26 ± 0.04 | 560 ± 25 | 641 ± 24 | 0.16 ± 0.10 |
| 0.5 | 1.27 ± 0.05 | 563 ± 25 | 649 ± 25 | 0.15 ± 0.10 |
| 1.0 | 1.29 ± 0.04 | 569 ± 25 | 654 ± 25 | 0.13 ± 0.10 |

### Mother Faces  
| Alpha | WH Ratio | Jaw (px) | Cheek (px) | ArcFace |
|-------|----------|----------|------------|---------|
| 0.0 | 1.42 ± 0.08 | 534 ± 55 | 617 ± 58 | 0.27 ± 0.16 |
| 0.5 | 1.41 ± 0.08 | 536 ± 55 | 620 ± 59 | 0.28 ± 0.14 |
| 1.0 | 1.42 ± 0.08 | 539 ± 56 | 624 ± 59 | 0.25 ± 0.14 |

## Key Observations

1. **No systematic trend with alpha**: WH ratio, jaw width, cheek width show no monotonic relationship with alpha
2. **Identity collapse**: ArcFace similarity < 0.6 for all conditions (acceptable is > 0.75)
3. **Child_avg is random noise**: The synthetic child latent_avg produces geometrically meaningless results

## Conclusion for Experiment 1

**H0 CANNOT BE TESTED** with current methodology because:
- The synthetic child latent_avg is not a valid child face representation
- It's just the mean of random latents on the adult manifold
- No meaningful interpolation can occur between two points on the same adult manifold

## Required Fix for Valid Alpha Sweep

A valid child latent_avg requires **one of**:
1. **Real child dataset** (CACD, MORPH, AgeDB) - encode with e4e, average
2. **Age regression** (InterFaceGAN age direction) - apply to adult latents, then average
3. **StyleGAN3 + age conditioning** - generate children, invert, average

## Next Steps

The alpha sweep methodology is valid but the child_avg construction must be fixed. Proceeding to Experiments 2-5 with current models to gather evidence about encoder residual vs latent_avg contributions.

---

**Confidence in H0 test**: 0.15 (methodology invalid, not a test of hypothesis)
**Evidence quality**: LOW - child_avg invalid
**Recommendation**: Fix child_avg construction before re-running alpha sweep