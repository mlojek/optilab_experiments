# Interpolation Criterion: Proposed Approach

## The Goal

We want to measure: **for each newly generated point in a CMA-ES generation, could we have
predicted its function value by interpolating from points we've already evaluated?**

This matters because if many new points fall in already-explored regions, a surrogate model
could have predicted their values — meaning those function evaluations were "wasted" and we
didn't actually need to call the expensive objective function.

---

## Why the Distribution-Based Approach Failed

Our first attempt checked whether a new point `x` had a high PDF value under any previous
generation's probability distribution. This failed because:

1. **Raw PDF** — In 10 dimensions, the raw probability density at any point is astronomically
   small (around 10⁻²⁴), always below any reasonable threshold. This gave 0% interpolation.

2. **Normalised PDF (ratio to peak)** — We fixed the scale issue by using `exp(−½·maha²)`
   instead, which equals 1.0 at the mean and 0.0 far away. But this still gave ~100%
   interpolation, because IPOP-CMA-ES starts with `sigma0 = 100` — a distribution so wide
   it covers the entire `[−100, 100]^10` search space. Once generation 0 is in history,
   almost every future point is "within" it.

**The fundamental problem:** distributions describe where points *could* be drawn from.
They don't tell us whether we actually evaluated points nearby. For interpolation, we need
*real evaluated points* to be close — not just a distribution that covered the region.

---

## The Proposed Solution: Point-Cloud Distance

Instead of asking *"was this region covered by a previous distribution?"*, we ask directly:

> **Is there a previously evaluated point close enough to x that we could accurately
> interpolate the function value at x from it?**

### What "close enough" means

In regular Euclidean distance, "close" is hard to define without knowing the scale of the
problem. A distance of 10 is huge when the algorithm has converged to a tiny region, but
tiny at the start when it's exploring a 200-unit-wide space.

The solution is to measure distance **relative to how spread out the current generation is**.
CMA-ES tracks its own scale via the covariance matrix `Σ = sigma² · C`, which describes the
shape and size of the cloud of points it's currently sampling. We use this to define a
**scaled distance** called the Mahalanobis distance.

---

## The Mahalanobis Distance (Explained Simply)

Imagine the current CMA-ES generation is sampling points from an ellipse (in 2D) or an
ellipsoid (in higher dimensions). The Mahalanobis distance measures *how many "ellipsoid
radii" away* a point is from the center.

```
Regular distance:   how far in absolute units (e.g. metres)
Mahalanobis distance: how far relative to the current sampling spread
```

Formally, the squared Mahalanobis distance between two points `x` and `y`, using the
current generation's covariance `Σ`, is:

```
maha²(x, y) = (x − y)ᵀ · Σ⁻¹ · (x − y)
```

Where `Σ⁻¹` is the **precision matrix** (inverse of the covariance matrix). Don't worry
about the formula — the key intuition is:

- `maha = 0` → `x` and `y` are the same point
- `maha = 1` → `y` is exactly 1 "standard deviation" away from `x` in the direction of the
  current sampling spread
- `maha = 2` → 2 standard deviations away
- `maha > 3` → quite far away relative to the current search scale

### Why this adapts automatically

At the start of optimisation, `sigma ≈ 100` and the sampling spread is huge. A Mahalanobis
distance of 1 corresponds to ~100 units in Euclidean space — so you'd need a previous point
very close in absolute terms to be considered "near".

As CMA-ES converges, `sigma` shrinks to e.g. `0.01`. Now Mahalanobis distance of 1
corresponds to only `0.01` Euclidean units. The criterion scales with the algorithm's
convergence automatically — we never need to tune it for the problem scale.

---

## The Criterion

A new point `x` in generation `g` is **interpolatable** if:

```
min over all previously evaluated points x_i:
    maha(x, x_i)  <  τ
```

Using the **current generation's covariance** `Σ_g` to compute the Mahalanobis distance.

`τ` is the threshold (default suggestion: `τ = 1.0`).

### Intuition for threshold values

| τ   | Meaning                                                                 |
|-----|-------------------------------------------------------------------------|
| 0.5 | A previous point is very close — within half a std dev. Excellent interpolation. |
| 1.0 | Within 1 std dev. Good interpolation likely.                           |
| 2.0 | Within 2 std devs. Reasonable interpolation.                           |
| 3.0 | Within 3 std devs. Marginal — the function may have changed a lot.     |

`τ = 1.0` is a natural default: it means "a previous point is as close as a typical
nearest-neighbour within the same generation would be."

---

## What This Fixes

| Problem                        | Distribution approach | Point-cloud approach |
|-------------------------------|----------------------|----------------------|
| Initial sigma0=100 covers all | ❌ ~100% always      | ✅ No previous points exist → 0% |
| Gen 0 is always 0%            | ✅                   | ✅                   |
| Scale-invariant                | ❌ threshold arbitrary | ✅ Mahalanobis adapts |
| Measures actual nearby points | ❌ measures distributions | ✅                |
| Computationally cheap         | ❌ matrix inversion per snapshot | ✅ one precision matrix per generation, dot products per point |

---

## Implementation Sketch

```python
@dataclass
class GenerationSnapshot:
    points: np.ndarray     # shape (pop_size, dim) — the evaluated x vectors
    precision: np.ndarray  # Σ⁻¹ of the CURRENT generation, shape (dim, dim)

def is_interpolatable(x, all_previous_points, current_precision, tau):
    """
    x:                    new point to classify, shape (dim,)
    all_previous_points:  array of all previously evaluated x, shape (N, dim)
    current_precision:    Σ⁻¹ of the current generation, shape (dim, dim)
    tau:                  Mahalanobis distance threshold
    """
    if len(all_previous_points) == 0:
        return False  # Generation 0: nothing to interpolate from

    diffs = all_previous_points - x           # shape (N, dim)
    maha2 = np.einsum('ni,ij,nj->n',
                      diffs, current_precision, diffs)  # shape (N,)
    return np.min(maha2) < tau ** 2
```

`np.einsum` computes all Mahalanobis distances at once efficiently using the pre-computed
precision matrix.

---

## Summary

| Step | What happens |
|------|-------------|
| Before gen 0 | No previous points. All points classified as extrapolation. |
| After gen 0 | Store the 7–14 evaluated `(x, y)` pairs. |
| Gen 1 | For each new point, compute Mahalanobis distance (using gen 1's `Σ`) to every point evaluated in gen 0. If min distance < τ → interpolatable. |
| Gen g | Same, but checking against ALL points from gens 0 through g−1. |

The result is a per-generation interpolation rate that starts at 0%, rises as CMA-ES
converges and begins resampling familiar territory, and reflects whether the function
evaluations in that generation were genuinely exploring new ground.
