"""Numerical core of AnalogExplorer.

Pure functions on numpy arrays: no file access, no Streamlit, no pandas.
Everything here must stay testable with small synthetic arrays.

Conventions
-----------
n  number of Earth tiles          K  number of variables
L  number of quantile levels      M  number of candidate sites

* Tile statistics are quantile vectors of shape (n, K, L).
* Similarities are handled in *log space* (log_sim <= 0) so that very poor
  matches do not underflow:  log s_tk = -0.5 * (W1_tk / sigma_k) ** 2, made
  linear beyond config.HUBER_DELTA tolerances (see log_similarity)
* The site score is a weighted geometric mean (non compensatory):
  score = exp(sum_k w_k * log s_tk)
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binom, rankdata

from src import config

EARTH_RADIUS_KM = 6371.0088


# ============================================================ similarity (3.5)
def wasserstein1(q_tiles: np.ndarray, q_target: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Mean absolute quantile gap: 1-D Wasserstein-1 distance between distributions.

    q_tiles: (n, K, L), q_target: (K, L)  ->  (n, K) float32.
    Processed in chunks to bound memory on large tile sets.
    """
    q_tiles = np.asarray(q_tiles)
    q_target = np.asarray(q_target)
    if q_tiles.ndim != 3 or q_target.ndim != 2 or q_tiles.shape[1:] != q_target.shape:
        raise ValueError(
            f"expected q_tiles (n, K, L) and q_target (K, L), got {q_tiles.shape} and {q_target.shape}")
    n, k, _ = q_tiles.shape
    out = np.empty((n, k), dtype=np.float32)
    tgt = q_target.astype(np.float32, copy=False)
    for start in range(0, n, chunk):
        out[start:start + chunk] = np.abs(q_tiles[start:start + chunk] - tgt).mean(axis=2)
    return out


def land_area_weights(lat_deg: np.ndarray, land_frac: np.ndarray | None = None) -> np.ndarray:
    """(n,) share of the land area held by each tile of a latitude/longitude grid; sums to 1.

    A 1 x 1 degree tile covers an area proportional to cos(latitude), times its land fraction. Counting
    tiles instead gives the polar regions several times their real share (29 % of the tiles of the real
    dataset lie south of 60 S, which holds about 9 % of Earth's land).
    """
    w = np.cos(np.radians(np.asarray(lat_deg, dtype=np.float64)))
    if land_frac is not None:
        w = w * np.asarray(land_frac, dtype=np.float64)
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        raise ValueError("tiles hold no land area")
    return w / w.sum()


def earth_pair_gaps(q_tiles: np.ndarray, n_pairs: int = 20_000, seed: int = 0,
                    weights: np.ndarray | None = None) -> np.ndarray:
    """W1 between random pairs of Earth tiles, per variable: (pairs, K) float32.

    This is the reference distribution of 'how different two places on Earth are',
    used for the default tolerance, the reachability test and absolute percentiles.
    weights: (n,) probability of drawing each tile (land_area_weights), so that a pair is two random
    places on Earth's land rather than two random tiles; None draws the tiles uniformly.
    """
    n = q_tiles.shape[0]
    if n < 2:
        raise ValueError("need at least two tiles to compare pairs")
    rng = np.random.default_rng(seed)
    if weights is None:
        i = rng.integers(0, n, n_pairs)
        j = rng.integers(0, n, n_pairs)
    else:
        p = np.asarray(weights, dtype=np.float64)
        i, j = rng.choice(n, n_pairs, p=p / p.sum()), rng.choice(n, n_pairs, p=p / p.sum())
    keep = i != j
    return np.abs(q_tiles[i[keep]] - q_tiles[j[keep]]).mean(axis=2).astype(np.float32)


def default_tolerances(q_tiles: np.ndarray, n_pairs: int = 20_000, seed: int = 0,
                       floor: float = 1e-6, weights: np.ndarray | None = None) -> np.ndarray:
    """Default tolerance sigma_k: typical W1 between two random places on Earth, per variable.

    A tile counts as "similar" when it is closer to the target than two random
    places on Earth are to each other. Mission-specific tolerances should
    override this when documented. sigma_k = P(the two places differ) x median gap of the pairs that
    differ. For a continuous variable every pair differs and this is the median gap. For a
    bounded variable many pairs are identical (two places without any bare ground, two places never frozen):
    the plain median is then 0, or a value just above 0 that jumps when the share of identical pairs crosses
    one half (bare ground: 0.18 with 51 % of identical pairs, 0.004 with 43 %). The product is the expected
    gap with a median in place of the mean of the differing pairs: no jump, and still robust to long tails.
    weights: see earth_pair_gaps.
    """
    gaps = earth_pair_gaps(q_tiles, n_pairs, seed, weights)
    typical = np.zeros(gaps.shape[1])
    for k in range(gaps.shape[1]):
        differ = gaps[:, k] > 0
        if differ.any():
            typical[k] = differ.mean() * np.median(gaps[differ, k])
    return np.maximum(typical, floor).astype(np.float32)


@dataclass(frozen=True)
class Reachability:
    """Can Earth match the target on each variable at all? (type-1 novelty, as in MESS / ExDet)."""

    best_w: np.ndarray        # (K,) smallest W1 over all Earth tiles (NaN = not evaluated)
    best_tile: np.ndarray     # (K,) tile reaching it
    percentile: np.ndarray    # (K,) share of Earth-Earth gaps smaller than best_w
    gap_sigma: np.ndarray     # (K,) best_w / sigma
    reachable: np.ndarray     # (K,) bool, False when best_w is beyond the max_pct Earth-Earth gap


def reachability(w1: np.ndarray, pair_gaps: np.ndarray, sigma: np.ndarray, max_pct: float = 0.99) -> Reachability:
    """Locate the closest Earth tile of each variable inside the Earth-Earth gap distribution.

    A variable whose best Earth tile is farther from the target than `max_pct` of the
    gaps between two random Earth places is out of Earth's range: ranking on it would
    only measure who is least bad, so it must leave the score and go to the limits report.
    """
    w1 = np.asarray(w1)
    k = w1.shape[1]
    evaluated = ~np.isnan(w1).all(axis=0)
    best_tile = np.zeros(k, dtype=np.int64)
    best_tile[evaluated] = np.nanargmin(w1[:, evaluated], axis=0)
    best_w = np.where(evaluated, w1[best_tile, np.arange(k)], np.nan)
    pct = np.array([np.mean(pair_gaps[:, j] < best_w[j]) if evaluated[j] else np.nan for j in range(k)])
    limit = np.quantile(pair_gaps, max_pct, axis=0)
    return Reachability(best_w, best_tile, pct, best_w / np.asarray(sigma), ~(best_w > limit))


def closer_than_share(w1_row: np.ndarray, pair_gaps: np.ndarray) -> np.ndarray:
    """Per variable, share of random Earth-Earth pairs farther apart than this site is from the target.

    (K,) in [0, 1], NaN when the variable is not evaluated. 0.5 = as close as two typical places on Earth.
    The reference is the pairs that differ on the variable: pairs identical on it (gap 0, e.g.
    two fully forested places on bare ground, two places frozen all year) come from the floor or ceiling
    of a bounded variable, not from a typical difference, and would make a perfect match look weak.
    For a continuous variable no pair is identical and the share is unchanged; a perfect match gives 1.

    One yardstick for the score and for the status: the site's gap is first divided by the share of
    pairs that differ. The tolerance of the score counts identical pairs as gaps of zero (default_tolerances:
    sigma = P(differ) x median gap of the differing pairs), so one tolerance corresponds to the median
    of the differing pairs: a site one tolerance away from the target reads 50 % for every variable, as it
    already did for a continuous one. Without the scaling a site frozen a quarter of the year read 'closer than
    92 %' to a permanently frozen target while its similarity was 0.10.
    """
    w = np.asarray(w1_row, dtype=np.float64)
    differ = (pair_gaps > 0).mean(axis=0)
    scaled = np.divide(w, differ, out=np.full(w.shape, np.inf), where=differ > 0)
    share = np.divide((pair_gaps > scaled[None, :]).mean(axis=0), differ, out=np.ones(w.shape), where=differ > 0)
    return np.where(np.isnan(w), np.nan, np.minimum(share, 1.0))


def similarity_at_closer_share(pair_gaps: np.ndarray, sigma: np.ndarray, share: float) -> np.ndarray:
    """Per variable, the similarity of a site that is closer to the target than exactly `share` of the Earth
    pairs that differ on it: the inverse of closer_than_share, (K,) in (0, 1].

    The statuses are thresholds on that share, so the matching similarity differs from one
    variable to the next; a chart of similarities must draw the threshold per variable, not as one circle.
    """
    gaps = np.asarray(pair_gaps, dtype=np.float64)
    differ = (gaps > 0).mean(axis=0)
    level = np.clip(1.0 - share * differ, 0.0, 1.0)
    reference = np.array([np.quantile(gaps[:, k], level[k]) for k in range(gaps.shape[1])])
    w = (reference * differ).astype(np.float32)             # the site's gap is read scaled by 1 / P(differ)
    return similarity(log_similarity(w, np.asarray(sigma, dtype=np.float32)))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values, for several hypotheses tested together; NaN stays NaN.

    A hypothesis is rejected at level alpha, with a family-wise error below alpha, when its adjusted p < alpha
   .
    """
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok], kind="stable")]
    m = order.size
    out[order] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    return out


def earth_percentile(log_score_values: np.ndarray, pair_gaps: np.ndarray, sigma: np.ndarray,
                     weights: np.ndarray) -> np.ndarray:
    """Absolute analog quality: share of random Earth-Earth pairs less similar than each site is to the target.

    0.5 = as close to the target as two typical places on Earth are to each other.
    """
    pair_scores = np.sort(log_score(log_similarity(pair_gaps, sigma), weights))
    return np.searchsorted(pair_scores, np.asarray(log_score_values), side="left") / pair_scores.size


def log_similarity(w1: np.ndarray, sigma: np.ndarray, delta: float | None = config.HUBER_DELTA) -> np.ndarray:
    """Log-similarity from a gap in tolerances z = W1 / sigma; same shape as w1, always <= 0.

    Huber penalty: log s = -z^2 / 2 up to z = delta, then
    -(delta * z - delta^2 / 2), which continues it with the same value and slope. Close matches
    are scored exactly as with a Gaussian; a very poor variable still costs, but linearly, so it
    no longer decides alone the order of the sites. delta=None keeps the quadratic everywhere.
    """
    z = w1 / np.asarray(sigma, dtype=w1.dtype)
    quadratic = -0.5 * z * z
    if delta is None:
        return quadratic
    d = z.dtype.type(delta)
    return np.where(z <= d, quadratic, d * (0.5 * d - z))             # NaN (not evaluated) stays NaN


def similarity(log_sim: np.ndarray) -> np.ndarray:
    """Similarity in (0, 1] from log-similarity."""
    return np.exp(log_sim)


# ==================================================================== weights
def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Scale non-negative weights so they sum to 1."""
    w = np.asarray(weights, dtype=np.float64)
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return w / total


def available_variables(q_target: np.ndarray) -> np.ndarray:
    """(K,) bool: variables measured for a target (a NaN row means 'not evaluated')."""
    return np.isfinite(np.asarray(q_target)).all(axis=-1)


def variable_weights_from_dimensions(dim_weights: Mapping[str, float], var_dimensions: Sequence[str],
                                     available: np.ndarray | None = None) -> np.ndarray:
    """Spread each dimension weight equally over its available variables, then normalise.

    A variable that is not available for the target gets a zero weight; its share goes
    to the other variables of the same dimension. A dimension with no available
    variable is dropped. Falls back to equal weights on the available variables when
    nothing positive remains.
    """
    avail = np.ones(len(var_dimensions), dtype=bool) if available is None else np.asarray(available, dtype=bool)
    counts = Counter(d for d, a in zip(var_dimensions, avail, strict=True) if a)
    raw = np.array([dim_weights.get(d, 0.0) / counts[d] if a else 0.0
                    for d, a in zip(var_dimensions, avail, strict=True)])
    if raw.sum() <= 0:
        raw = avail.astype(np.float64)
    return normalize_weights(raw)


# ================================================================ score (4.2)
def log_score(log_sim: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted geometric mean in log space: (n, K) @ (K,) -> (n,).

    A zero weight ignores a variable, even when its similarity is NaN (not evaluated).
    """
    w = np.asarray(weights, dtype=log_sim.dtype)
    return np.where(w > 0, log_sim, 0) @ w


def dimension_log_similarity(log_sim: np.ndarray, var_dimensions: Sequence[str],
                             dimensions: Sequence[str]) -> np.ndarray:
    """Mean log-similarity per dimension (geometric mean inside the dimension): (..., K) -> (..., D).

    Variables that are not evaluated (NaN) are skipped; a dimension with none left is NaN.
    """
    out = []
    for dim in dimensions:
        sub = log_sim[..., [i for i, d in enumerate(var_dimensions) if d == dim]]
        n = np.isfinite(sub).sum(axis=-1)
        total = np.where(np.isfinite(sub), sub, 0).sum(axis=-1)
        out.append(np.where(n > 0, total / np.maximum(n, 1), np.nan))
    return np.stack(out, axis=-1)


# ========================================================= geometry (4.3, 4.4)
def unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Points on the unit sphere, shape (n, 3). Euclidean chord distance is monotone with great-circle distance."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    cl = np.cos(lat)
    return np.column_stack((cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)))


def build_tree(xyz: np.ndarray) -> cKDTree:
    """KD-tree over unit-sphere points (build once per dataset)."""
    return cKDTree(xyz)


def chord_to_km(chord: np.ndarray) -> np.ndarray:
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(np.asarray(chord) / 2.0, 0.0, 1.0))


def km_to_chord(km: float) -> float:
    return 2.0 * np.sin(km / (2.0 * EARTH_RADIUS_KM))


def haversine_km(lat1, lon1, lat2, lon2, radius_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Great-circle distance in km (broadcasting) on a sphere of `radius_km` (Earth by default)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nearest_distance_km(lat: np.ndarray, lon: np.ndarray, ref_lat: np.ndarray,
                        ref_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance (km) and index of the nearest reference point, for each point. O(n log m)."""
    tree = cKDTree(unit_xyz(ref_lat, ref_lon))
    chord, idx = tree.query(unit_xyz(lat, lon))
    return chord_to_km(chord), idx


def novelty(distance_km: np.ndarray, lam_km: float) -> np.ndarray:
    """N = 1 - exp(-d / lambda), d = distance to the nearest catalogued analog site."""
    return 1.0 - np.exp(-np.asarray(distance_km) / lam_km)


def understudied(lat: np.ndarray, lon: np.ndarray, known_dist_km: np.ndarray, min_km: float,
                 regions: Mapping[str, tuple[float, float, float, float]]) -> np.ndarray:
    """(n,) bool: tiles at least `min_km` from every catalogued analog and outside famous regions."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    famous = np.zeros(lat.shape, dtype=bool)
    for lat_min, lat_max, lon_min, lon_max in regions.values():
        famous |= (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    return (np.asarray(known_dist_km) >= min_km) & ~famous


def non_max_suppression(score: np.ndarray, xyz: np.ndarray, radius_km: float, max_sites: int,
                        tree: cKDTree | None = None) -> np.ndarray:
    """Greedy distinct sites: keep the best tile, drop tiles within R km, repeat (4.4).

    Returns tile indices ordered from best to worst. Non-finite scores are never picked.
    """
    n = score.shape[0]
    tree = tree if tree is not None else cKDTree(xyz)
    order = np.argsort(-np.where(np.isfinite(score), score, -np.inf), kind="stable")
    radius = km_to_chord(radius_km)
    blocked = ~np.isfinite(score)
    picked: list[int] = []
    for i in order:
        if blocked[i]:
            continue
        picked.append(int(i))
        if len(picked) >= max_sites:
            break
        blocked[tree.query_ball_point(xyz[i], radius)] = True
    return np.asarray(picked, dtype=np.int64) if n else np.empty(0, dtype=np.int64)


# ============================================== similarity vs novelty (4.3)
def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Rank of each value scaled to (0, 1]; ties share their average rank."""
    return rankdata(x, method="average") / len(x)


def blend_rank(log_score_: np.ndarray, novelty_: np.ndarray, novelty_mix: float) -> np.ndarray:
    """Position on the similarity/novelty trade-off (0 = similarity only, 1 = novelty only).

    Uses percentile ranks so both axes are comparable. mix=0 returns the log-score
    itself (same ordering, no extra sort).
    """
    if novelty_mix <= 0:
        return log_score_
    if novelty_mix >= 1:
        return novelty_
    return (1 - novelty_mix) * percentile_rank(log_score_) + novelty_mix * percentile_rank(novelty_)


def pareto_front(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points when maximising both a and b. O(n log n).

    Exact duplicates collapse to their first occurrence.
    """
    n = a.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    order = np.lexsort((-b, -a))  # primary key: a descending, then b descending
    b_sorted = b[order]
    running_max = np.maximum.accumulate(b_sorted)
    keep = np.ones(n, dtype=bool)
    keep[1:] = b_sorted[1:] > running_max[:-1]
    mask[order[keep]] = True
    return mask


# ========================================================== ranking pipeline
@dataclass(frozen=True)
class Ranking:
    log_sim: np.ndarray    # (n, K)
    log_score: np.ndarray  # (n,)
    blended: np.ndarray    # (n,) ordering key used for site selection
    site_idx: np.ndarray   # (<= max_sites,) distinct best sites


def rank_sites(w1: np.ndarray, sigma: np.ndarray, weights: np.ndarray, novelty_values: np.ndarray,
               novelty_mix: float, xyz: np.ndarray, tree: cKDTree, radius_km: float,
               max_sites: int) -> Ranking:
    """From per-variable distances to a list of distinct best sites. Cheap enough to re-run on every slider move."""
    ls_sim = log_similarity(w1, sigma)
    ls = log_score(ls_sim, weights)
    blended = blend_rank(ls, novelty_values, novelty_mix)
    sites = non_max_suppression(blended, xyz, radius_km, max_sites, tree)
    return Ranking(ls_sim, ls, blended, sites)


def top_fraction(score: np.ndarray, idx: np.ndarray, weights: np.ndarray | None = None,
                 ties: str = "worst") -> np.ndarray:
    """Share of the tiles scoring at least as well as each tile in idx (0.01 = top 1 %).

    weights: (n,) land area of each tile (land_area_weights): the result is then a share of Earth's land,
    not of the tiles. ties: "worst" counts the tiles tied with it as ahead (the share scoring at least as
    well); "mid" counts half of them (average rank of the tie, as rank statistics do), so that a random
    place ranks 'top 50 %' on average whatever the ties.
    """
    if ties not in ("worst", "mid"):
        raise ValueError(f"unknown tie rule {ties!r}")
    order = np.argsort(score, kind="stable")
    srt = score[order]
    w = np.ones(score.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(w[order])])
    worse = cum[np.searchsorted(srt, score[idx], side="left")]          # strictly worse
    upto = cum[np.searchsorted(srt, score[idx], side="right")]          # worse or tied (itself included)
    ahead = cum[-1] - (worse if ties == "worst" else 0.5 * (worse + upto))
    return ahead / cum[-1]


# ============================================================ robustness (4.5)
def _dirichlet_draws(weights: np.ndarray, alpha: float | Sequence[float], n_draws: int,
                     rng: np.random.Generator, drop_prob: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Perturbed weights over the active (w > 0) variables.

    Each draw follows Dirichlet(a * w), with the concentration a picked among `alpha` (one value
    or several levels of uncertainty on the weights). With probability `drop_prob` one active
    variable is also left out (weight 0, the others renormalised): structural uncertainty on
    which variables matter (OECD/JRC Handbook on composite indicators, 2008).
    Returns (active variable indices, draws (n_draws, n_active) float32, rows summing to 1).
    """
    w = np.asarray(weights, dtype=np.float64)
    active = np.flatnonzero(w > 0)
    if active.size == 0:
        raise ValueError("at least one weight must be positive")
    alphas = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    a = alphas[rng.integers(0, alphas.size, n_draws)]
    g = rng.standard_gamma(a[:, None] * w[active][None, :])            # Dirichlet = normalised gammas
    if drop_prob > 0 and active.size > 1:
        drop = np.flatnonzero(rng.random(n_draws) < drop_prob)
        g[drop, rng.integers(0, active.size, drop.size)] = 0.0
    g = np.maximum(g, 1e-300)
    return active, (g / g.sum(axis=1, keepdims=True)).astype(np.float32)


def score_interval(w1: np.ndarray, sigma: np.ndarray, weights: np.ndarray, idx: np.ndarray,
                   alpha: float | Sequence[float] = 50.0, n_draws: int = 300,
                   q: tuple[float, float] = (0.05, 0.95), seed: int = 0, drop_prob: float = 0.0,
                   relative: bool = False) -> np.ndarray:
    """Interval of the score of selected tiles when the weights are perturbed.

    Same draws as `robustness` (same seed and settings). Returns (len(idx), 2) scores in (0, 1]: lower
    and upper quantiles `q`. relative=False: absolute scores, evaluated only on the tiles in `idx`.
    relative=True: in every draw the score is divided by the best tile's score of that same draw, which
    is the relative score the app shows (1 = best Earth tile), so the range can never exceed 1. Dividing
    an absolute range by the unperturbed best score, as the app first did, gave ranges up to 2.9.
    """
    rng = np.random.default_rng(seed)
    active, draws = _dirichlet_draws(weights, alpha, n_draws, rng, drop_prob)
    sig = np.asarray(sigma, dtype=np.float32)[active]
    idx = np.asarray(idx, dtype=np.int64)
    if relative:
        log_scores = log_similarity(np.asarray(w1, dtype=np.float32)[:, active], sig) @ draws.T   # (n, n_draws)
        scores = np.exp(log_scores[idx] - log_scores.max(axis=0))
    else:
        scores = np.exp(log_similarity(np.asarray(w1, dtype=np.float32)[idx][:, active], sig) @ draws.T)
    return np.quantile(scores, q, axis=1).T


def robustness(w1: np.ndarray, sigma: np.ndarray, weights: np.ndarray, alpha: float | Sequence[float] = 50.0,
               n_draws: int = 300, top_frac: float = 0.01, sigma_rel_std: float = 0.0,
               seed: int = 0, chunk_draws: int = 32, drop_prob: float = 0.0,
               tile_weights: np.ndarray | None = None) -> np.ndarray:
    """Probability that each tile stays in the top fraction when the weights (and, with drop_prob,
    the set of variables) are perturbed; see _dirichlet_draws. Optionally the tolerances are
    perturbed by a log-normal factor. Returns an (n,) array in [0, 1].
    tile_weights: (n,) land area of each tile; the top fraction is then a share of Earth's land (the best
    tiles holding `top_frac` of the area), like the 'Top %' shown next to it, instead of a share of tiles.
    """
    if tile_weights is not None:
        if sigma_rel_std > 0:
            raise ValueError("tile_weights cannot be combined with perturbed tolerances")
        return _robustness_by_area(w1, sigma, weights, alpha, n_draws, top_frac, seed, chunk_draws, drop_prob,
                                   tile_weights)
    n = w1.shape[0]
    rng = np.random.default_rng(seed)
    active, draws = _dirichlet_draws(weights, alpha, n_draws, rng, drop_prob)
    sig = np.asarray(sigma, dtype=np.float32)[active]
    w1a = np.ascontiguousarray(w1[:, active], dtype=np.float32)
    k_top = max(int(np.ceil(top_frac * n)), 1)
    hits = np.zeros(n, dtype=np.int32)
    if sigma_rel_std <= 0:
        log_sim = log_similarity(w1a, sig)
        for s in range(0, n_draws, chunk_draws):
            scores = log_sim @ draws[s:s + chunk_draws].T                 # (n, d)
            thr = np.partition(scores, n - k_top, axis=0)[n - k_top]      # k-th largest per draw
            hits += (scores >= thr).sum(axis=1, dtype=np.int32)
    else:
        mult = rng.lognormal(0.0, sigma_rel_std, size=(n_draws, active.size)).astype(np.float32)
        for d in range(n_draws):
            scores = log_similarity(w1a, sig * mult[d]) @ draws[d]
            thr = np.partition(scores, n - k_top)[n - k_top]
            hits += scores >= thr
    return hits / n_draws


def _robustness_by_area(w1, sigma, weights, alpha, n_draws, top_frac, seed, chunk_draws, drop_prob,
                        tile_weights) -> np.ndarray:
    """robustness() when the top fraction is a share of the land area: same draws, same scores."""
    n = w1.shape[0]
    rng = np.random.default_rng(seed)
    active, draws = _dirichlet_draws(weights, alpha, n_draws, rng, drop_prob)
    log_sim = log_similarity(np.ascontiguousarray(w1[:, active], dtype=np.float32),
                             np.asarray(sigma, dtype=np.float32)[active])
    area = np.asarray(tile_weights, dtype=np.float64)
    area = area / area.sum()
    hits = np.zeros(n, dtype=np.int32)
    for s in range(0, n_draws, chunk_draws):
        scores = log_sim @ draws[s:s + chunk_draws].T                     # (n, d)
        order = np.argsort(-scores, axis=0, kind="stable")                # best first, per draw
        before = np.cumsum(area[order], axis=0) - area[order]             # area ranked ahead of each tile
        for d in range(order.shape[1]):
            hits[order[before[:, d] < top_frac - 1e-9, d]] += 1           # the best tile always counts
    return hits / n_draws


# ================================================ complementary ensemble (4.6)
@dataclass(frozen=True)
class EnsembleResult:
    indices: tuple[int, ...]          # rows of the candidate matrix
    coverage: float                   # C(S) = sum_k w_k max_{j in S} s_jk
    ceiling: float                    # best-per-variable upper bound (one site per variable)
    best_single: float                # best coverage reached by a single site
    best_per_variable: np.ndarray     # (K,) max similarity reached inside the set
    best_site_per_variable: np.ndarray  # (K,) position inside `indices` of the best site
    uncovered: np.ndarray             # (K,) bool, best similarity below the covered threshold
    feasible: bool                    # False if the critical-variable constraint could not be met
    method: str                       # "exhaustive" or "greedy"


def coverage(sim_sites: np.ndarray, weights: np.ndarray, subset: Sequence[int]) -> float:
    """C(S) = sum_k w_k * max_{j in S} s_jk."""
    return float(np.asarray(weights) @ np.asarray(sim_sites)[list(subset)].max(axis=0))


def candidate_pool(log_sim: np.ndarray, weights: np.ndarray, var_dimensions: Sequence[str],
                   xyz: np.ndarray, tree: cKDTree, radius_km: float, n_overall: int,
                   n_per_dimension: int) -> np.ndarray:
    """Candidate sites for the ensemble: best overall + best specialists of each weighted dimension.

    A site that is excellent on one dimension but poor on the others has a low
    geometric score and would never reach an overall top-M, yet it is exactly what
    a complementary set needs. Returns unique tile indices, overall ranking first.
    """
    weights = np.asarray(weights)
    parts = [non_max_suppression(log_score(log_sim, weights), xyz, radius_km, n_overall, tree)]
    for dim in dict.fromkeys(var_dimensions):
        cols = [i for i, d in enumerate(var_dimensions) if d == dim and weights[i] > 0]
        if cols:
            parts.append(non_max_suppression(log_sim[:, cols].mean(axis=1), xyz, radius_km,
                                             n_per_dimension, tree))
    return np.fromiter(dict.fromkeys(np.concatenate(parts).tolist()), dtype=np.int64)


def _best_exhaustive(S: np.ndarray, w: np.ndarray, k: int, tau: np.ndarray | None):
    """Exact search for k <= 3. Returns (best coverage, indices) or (-inf, None) if infeasible."""
    M = S.shape[0]
    best_val, best_idx = -np.inf, None
    if k == 1:
        cov = S @ w
        if tau is not None:
            cov = np.where((S >= tau).all(axis=1), cov, -np.inf)
        i = int(np.argmax(cov))
        return (float(cov[i]), (i,)) if np.isfinite(cov[i]) else (-np.inf, None)
    if k == 2:
        for i in range(M - 1):
            T = np.maximum(S[i], S[i + 1:])
            cov = T @ w
            if tau is not None:
                cov = np.where((T >= tau).all(axis=1), cov, -np.inf)
            j = int(np.argmax(cov))
            if cov[j] > best_val:
                best_val, best_idx = float(cov[j]), (i, i + 1 + j)
        return best_val, best_idx
    if k == 3:
        pair_max = np.maximum(S[:, None, :], S[None, :, :])   # (M, M, K), reused for every i
        upper = np.triu(np.ones((M, M), dtype=bool), 1)
        for i in range(M - 2):
            m = M - i - 1
            T = np.maximum(pair_max[i + 1:, i + 1:], S[i])      # (m, m, K)
            cov = T @ w
            ok = upper[:m, :m]
            if tau is not None:
                ok = ok & (T >= tau).all(axis=2)
            cov = np.where(ok, cov, -np.inf)
            flat = int(np.argmax(cov))
            if cov.flat[flat] > best_val:
                row, col = divmod(flat, m)
                best_val, best_idx = float(cov.flat[flat]), (i, i + 1 + row, i + 1 + col)
        return best_val, best_idx
    raise ValueError("exhaustive search supports k <= 3")


def _greedy(S: np.ndarray, w: np.ndarray, k: int) -> tuple[int, ...]:
    """Greedy maximisation; guarantees at least (1 - 1/e) of the optimum (monotone submodular)."""
    current = np.zeros(S.shape[1], dtype=S.dtype)
    chosen: list[int] = []
    for _ in range(k):
        gains = np.maximum(S, current) @ w
        gains[chosen] = -np.inf
        i = int(np.argmax(gains))
        chosen.append(i)
        current = np.maximum(current, S[i])
    return tuple(chosen)


def best_complementary_set(sim_sites: np.ndarray, weights: np.ndarray, k: int, *,
                           tau: np.ndarray | None = None, covered_threshold: float = 0.5,
                           method: str = "auto") -> EnsembleResult:
    """Best set of k sites maximising C(S) = sum_k w_k max_{j in S} s_jk.

    sim_sites: (M, K) similarities in (0, 1]; a NaN column is a variable not evaluated for
    the target (it must have a zero weight and is never reported as uncovered).
    tau: optional (K,) minimum coverage for critical variables (NaN = not critical); if
    no set satisfies it the unconstrained optimum is returned with feasible=False. The
    constraint is only enforced by the exhaustive search. method: "auto" | "exhaustive" | "greedy".
    """
    S = np.ascontiguousarray(sim_sites, dtype=np.float32)
    evaluated = np.isfinite(S).all(axis=0)
    S = np.where(evaluated, S, np.float32(0))
    w = np.asarray(weights, dtype=np.float32)
    M, K = S.shape
    if M == 0:
        raise ValueError("no candidate site")
    k = int(min(max(k, 1), M))
    tau_eff = None
    if tau is not None:
        tau_arr = np.asarray(tau, dtype=np.float64)
        tau_eff = np.where(np.isnan(tau_arr), -np.inf, tau_arr).astype(np.float32)
    if method == "auto":
        exhaustive_ok = k == 1 or (k == 2 and M <= 3000) or (k == 3 and M <= 400)
        method = "exhaustive" if exhaustive_ok else "greedy"

    feasible = True
    if method == "exhaustive":
        _, idx = _best_exhaustive(S, w, k, tau_eff)
        if idx is None:
            feasible = False
            _, idx = _best_exhaustive(S, w, k, None)
    elif method == "greedy":
        idx = _greedy(S, w, k)
        feasible = tau_eff is None or bool((S[list(idx)].max(axis=0) >= tau_eff).all())
    else:
        raise ValueError(f"unknown method {method!r}")

    chosen = S[list(idx)]
    best_var = chosen.max(axis=0)
    return EnsembleResult(
        indices=tuple(int(i) for i in idx),
        coverage=float(w @ best_var),
        ceiling=float(w @ S.max(axis=0)),
        best_single=float((S @ w).max()),
        best_per_variable=np.where(evaluated, best_var, np.nan),
        best_site_per_variable=chosen.argmax(axis=0),
        uncovered=(best_var < covered_threshold) & evaluated,
        feasible=feasible,
        method=method,
    )


def ensemble_curve(sim_sites: np.ndarray, weights: np.ndarray, k_max: int = 3,
                   **kwargs) -> list[EnsembleResult]:
    """Best ensemble for k = 1 .. k_max (coverage vs number of sites)."""
    return [best_complementary_set(sim_sites, weights, k, **kwargs) for k in range(1, k_max + 1)]


# ================================================= credibility check (4.7)
def nearest_tiles(tree: cKDTree, ref_lat: np.ndarray, ref_lon: np.ndarray,
                  max_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest tile of each reference point: (tile index, -1 if farther than max_km; distance km)."""
    chord, idx = tree.query(unit_xyz(ref_lat, ref_lon))
    dist = chord_to_km(chord)
    return np.where(dist <= max_km, idx, -1), dist


@dataclass(frozen=True)
class CredibilityTest:
    n_sites: int         # known analogs matched to a tile
    mean_rank: float     # mean top fraction of those sites (0 = best tile; compare with null_mean)
    null_mean: float     # the same mean for random places: 0.5 with mid-ranked ties (more with "worst" ties)
    p_value: float       # chance that random land tiles rank at least as well
    z: float             # effect size against the random draws
    n_in_top: int        # sites in the top `top_frac` (former criterion)
    binomial_p: float    # exact p-value of that count under random ranking
    conclusive: bool     # enough sites for the test to have power
    passed: bool


def credibility_test(score: np.ndarray, site_top_frac: np.ndarray, top_frac: float = 0.10, n_draws: int = 10_000,
                     alpha: float = 0.05, min_sites: int = 5, seed: int = 0) -> CredibilityTest:
    """Do known analogs rank better than random land tiles?

    Statistic: mean percentile rank of the known sites (Pearson et al. 2007 style), compared
    with `n_draws` sets of random tiles of the same size: p = (1 + #{null <= observed}) / (n_draws + 1).
    With fewer than `min_sites` sites the result is reported as inconclusive (no power).
    site_top_frac: top fraction of each known site (NaN = not matched, ignored).
    """
    vals = np.asarray(site_top_frac, dtype=np.float64)
    ranks = top_fraction(score, np.arange(score.shape[0]), ties="mid")
    return credibility_test_maps(vals, [ranks] * vals.size, top_frac, n_draws, alpha, min_sites, seed)


def credibility_test_maps(site_ranks: np.ndarray, rank_maps: Sequence[np.ndarray], top_frac: float = 0.10,
                          n_draws: int = 10_000, alpha: float = 0.05, min_sites: int = 5,
                          seed: int = 0, tile_weights: np.ndarray | None = None) -> CredibilityTest:
    """Same test when each known site is ranked on its own score map (e.g. the dimensions it is famous for).

    site_ranks: (m,) top fraction of each site in its map (NaN = not matched, ignored);
    rank_maps: m arrays (n,) of the top fraction of every tile in that site's map. The null
    draws one random tile per site in the site's own map, so the comparison stays fair.
    tile_weights: (n,) land area of each tile: the null then draws random places on land, not random tiles.
    """
    vals = np.asarray(site_ranks, dtype=np.float64)
    keep = np.flatnonzero(np.isfinite(vals))
    vals, maps = vals[keep], [rank_maps[i] for i in keep]
    m = int(vals.size)
    if m == 0:
        return CredibilityTest(0, np.nan, np.nan, np.nan, np.nan, 0, np.nan, False, False)
    rng = np.random.default_rng(seed)
    if tile_weights is None:
        null = np.mean([r[rng.integers(0, r.size, n_draws)] for r in maps], axis=0)
    else:
        p = np.asarray(tile_weights, dtype=np.float64)
        null = np.mean([r[rng.choice(r.size, n_draws, p=p / p.sum())] for r in maps], axis=0)
    observed = float(vals.mean())
    p = (1 + int((null <= observed).sum())) / (n_draws + 1)
    z = (observed - null.mean()) / max(null.std(), 1e-12)
    n_top = int((vals <= top_frac).sum())
    conclusive = m >= min_sites
    return CredibilityTest(m, observed, float(null.mean()), p, float(z), n_top,
                           float(binom.sf(n_top - 1, m, top_frac)), conclusive, bool(conclusive and p < alpha))


def self_similarity(q_target: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Sanity test: similarity of a target with itself, per variable. Must be exactly 1."""
    q = np.asarray(q_target, dtype=np.float32)
    return similarity(log_similarity(wasserstein1(q[None], q), sigma))[0]
