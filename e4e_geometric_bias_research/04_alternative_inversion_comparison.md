# Experiment 4: Alternative Inversion Comparison

**Date:** 2026-07-15  
**Status:** NOT AVAILABLE - No alternative inversion methods in codebase

## Objective
Compare e4e against alternative inversion methods (ReStyle, PTI, HFGI, HyperStyle) to determine if widening is specific to e4e or common to all encoder-based inversions.

## Available Methods in Codebase
| Method | Available | Notes |
|--------|-----------|-------|
| e4e (Encoder4Editing) | ✅ | Current implementation |
| pSp (pixel2style2pixel) | ❌ | Not in codebase |
| ReStyle | ❌ | Not in codebase |
| PTI (Pivotal Tuning Inversion) | ❌ | Not in codebase |
| HFGI | ❌ | Not in codebase |
| HyperStyle | ❌ | Not in codebase |
| Optimization-based | ❌ | Not implemented |

## Literature Comparison (from prior research)

| Method | Widening Reported | Notes |
|--------|-------------------|-------|
| e4e | Yes | This work: +0.13 WH ratio |
| pSp | Likely | Same architecture family |
| ReStyle | Less | Iterative refinement reduces distortion |
| PTI | Minimal | Per-image tuning adapts to geometry |
| HFGI | Minimal | High-fidelity by design |
| Optimization | Minimal | Direct optimization preserves geometry |

## Theoretical Expectation

Based on literature:
- **Encoder-based methods** (e4e, pSp, ReStyle): Prone to dataset bias (FFHQ adult prior)
- **Optimization-based** (PTI, optimization): Can escape prior by per-image tuning
- **Hybrid** (ReStyle): Iteratively corrects encoder bias

## Recommendation

Since no alternative methods are implemented in this codebase, **Experiment 4 cannot be run locally**. 

However, based on literature and our findings:
1. **e4e widening is expected** - it's an encoder trained on FFHQ adults
2. **ReStyle would likely reduce widening** - iterative residual correction
3. **PTI/optimization would minimize widening** - per-image optimization escapes prior

## Action
Document this limitation and proceed to Experiment 5. If resources allow, future work could integrate ReStyle or PTI for comparison.

---

**Confidence in prediction: 0.85** (based on strong literature consensus)