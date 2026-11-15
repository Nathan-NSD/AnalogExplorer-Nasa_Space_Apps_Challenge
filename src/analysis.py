"""Engine orchestration on a dataset: one place that ranks sites.

No Streamlit. Thin compositions of math_engine functions, shared by the app, the offline
steps of data_prep (fine-scale zoom) and the written atlas, so that all of them select and
score sites in exactly the same way.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src import config, data_prep, places
from src import math_engine as me

if TYPE_CHECKING:
    from src.data_prep import Dataset


@dataclass
class Context:
    """Dataset plus everything derived from it once (geometry, novelty, Earth reference)."""

    ds: Dataset
    xyz: np.ndarray
    tree: object
    known_dist_km: np.ndarray   # (n,) distance to the nearest catalogued analog
    known_idx: np.ndarray       # (n,) index of that analog
    novelty: np.ndarray         # (n,)
    pair_gaps: np.ndarray       # (pairs, K) W1 between two random places on land (reference distribution)
    understudied: np.ndarray    # (n,) bool, far from catalogued analogs and outside famous regions
    area: np.ndarray            # (n,) share of Earth's land area held by each tile
    place: np.ndarray           # (n,) display name of each tile ('' when unknown), from the places layer


def build_context(ds: Dataset) -> Context:
    """Geometry, novelty and Earth-Earth reference gaps of a dataset."""
    xyz = me.unit_xyz(ds.tile_lat, ds.tile_lon)
    dist, idx = me.nearest_distance_km(ds.tile_lat, ds.tile_lon,
                                       ds.known_analogs["lat"].to_numpy(), ds.known_analogs["lon"].to_numpy())
    area = me.land_area_weights(ds.tile_lat, ds.valid_frac)
    return Context(ds, xyz, me.build_tree(xyz), dist, idx, me.novelty(dist, config.NOVELTY_LAMBDA_KM),
                   me.earth_pair_gaps(ds.quantiles, weights=area),
                   me.understudied(ds.tile_lat, ds.tile_lon, dist, config.UNDERSTUDIED_MIN_KM, config.FAMOUS_REGIONS),
                   area, places.tile_labels(ds.places, ds.n_tiles))


def top_share(ctx: Context, score: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Share of Earth's land area scoring at least as well as each tile of idx (0.01 = top 1 % of the land)."""
    return me.top_fraction(score, np.asarray(idx), ctx.area)


def mission_dimension_weights(mission: str) -> dict[str, float]:
    """Numeric dimension weights of a mission profile."""
    return {d: config.LEVEL_VALUES[level] for d, level in config.MISSION_PROFILES[mission].items()}


def tolerances(ctx: Context, tol_scale: float = 1.0) -> np.ndarray:
    """(K,) tolerances: default sigma times the user's scale."""
    return ctx.ds.sigma_default * np.float32(tol_scale)


def target_reachability(ctx: Context, w1: np.ndarray) -> me.Reachability:
    """Which variables of a target Earth can match at all; independent of the weights."""
    return me.reachability(w1, ctx.pair_gaps, ctx.ds.sigma_default, config.REACHABLE_MAX_PCT)


def scoring_mask(q_target: np.ndarray, reach: me.Reachability) -> np.ndarray:
    """(K,) bool: variables that enter the score: measured for the target and within Earth's range."""
    available = me.available_variables(q_target)
    mask = available & reach.reachable
    return mask if mask.any() else available


@dataclass(frozen=True)
class TargetAnalysis:
    w1: np.ndarray            # (n, K) distances of every tile to the target
    reach: me.Reachability
    weights: np.ndarray       # (K,) variable weights, zero on variables left out of the score
    ranking: me.Ranking


def analyse_target(ctx: Context, t: int, dim_weights: Mapping[str, float], tol_scale: float = 1.0,
                   novelty_mix: float = 0.0, radius_km: float = config.DEFAULT_RADIUS_KM,
                   w1: np.ndarray | None = None, reach: me.Reachability | None = None) -> TargetAnalysis:
    """Rank the Earth tiles for target `t`.

    w1, reach: optional precomputed values (the app caches them per target).
    """
    q_target = ctx.ds.target_quantiles[t]
    w1 = me.wasserstein1(ctx.ds.quantiles, q_target) if w1 is None else w1
    reach = target_reachability(ctx, w1) if reach is None else reach
    weights = me.variable_weights_from_dimensions(dim_weights, config.VAR_DIMENSIONS, scoring_mask(q_target, reach))
    ranking = me.rank_sites(w1, tolerances(ctx, tol_scale), weights, ctx.novelty, novelty_mix, ctx.xyz, ctx.tree,
                            radius_km, config.POOL_OVERALL)
    return TargetAnalysis(w1, reach, weights, ranking)


def best_understudied(ctx: Context, sites: np.ndarray) -> int | None:
    """First distinct site that is not a well-known analog (the challenge asks to look beyond them)."""
    return next((int(i) for i in sites if ctx.understudied[i]), None)


def tile_zoom(ctx: Context, tile: int):
    """Stored fine-scale zoom (data_prep.TileZoom) of a tile, or None."""
    return ctx.ds.zoom.get(data_prep.zoom_key(ctx.ds.tile_lat[tile], ctx.ds.tile_lon[tile]))


# =================================================== credibility check (4.7)
@dataclass(frozen=True)
class Credibility:
    test: me.CredibilityTest
    props: list                   # (A,) documented properties each known analog is tested on (empty = not tested)
    dist_km: np.ndarray           # (A,) distance from each analog to its tile
    ranks: np.ndarray             # (A,) top fraction on its own properties (NaN = not tested or no tile)
    control: np.ndarray | None    # (A,) the same ranks for a target of the other body
    control_target: int | None
    self_ok: bool                 # sanity test: the target compared with itself gives s = 1
    # What explains each rank (report.credibility_table); NaN / -1 / False where the site is not ranked.
    tiles: np.ndarray             # (A,) tile of each analog, -1 = no land tile within reach
    worst_var: np.ndarray         # (A,) tested variable with the lowest similarity
    worst_sim: np.ndarray         # (A,) that similarity
    top_tie: np.ndarray           # (A,) share of the tiles tied at the best score of the site's properties
    at_top: np.ndarray            # (A,) bool, the site is one of them (exact match on what is tested)


@dataclass(frozen=True)
class PropertyRanks:
    ranks: np.ndarray             # (A,) top fraction of each site on its own properties (NaN = not ranked)
    maps: list                    # per site, the top fraction of every tile on the same properties
    worst_var: np.ndarray         # (A,) tested variable with the lowest similarity (-1 = not ranked)
    worst_sim: np.ndarray         # (A,) that similarity
    top_tie: np.ndarray           # (A,) share of the tiles tied at the best score
    at_top: np.ndarray            # (A,) bool, the site has that best score


def property_ranks(ctx: Context, t: int, props_per_site: list, tiles: np.ndarray, tol_scale: float = 1.0,
                   w1: np.ndarray | None = None) -> PropertyRanks:
    """Rank (top fraction) of each known site on the score of its own documented properties, for target t.

    Only the target's usable variables count (measured, within Earth's range); a site is not ranked when it
    is untested, has no tile or has nothing usable. The rank is a share of Earth's land area, and tiles that
    match the target exactly on a bounded variable (aridity, frozen ground) share the average rank of their
    tie; top_tie (land share of the tie at the best score) and at_top report it.
    """
    q_target = ctx.ds.target_quantiles[t]
    w1 = me.wasserstein1(ctx.ds.quantiles, q_target) if w1 is None else w1
    log_sim = me.log_similarity(w1, tolerances(ctx, tol_scale))
    usable = scoring_mask(q_target, target_reachability(ctx, w1))
    every_tile = np.arange(ctx.ds.n_tiles)
    maps: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    n = len(tiles)
    ranks, site_maps = np.full(n, np.nan), []
    worst_var, worst_sim = np.full(n, -1, dtype=np.int64), np.full(n, np.nan)
    top_tie, at_top = np.full(n, np.nan), np.zeros(n, dtype=bool)
    for i, (props, tile) in enumerate(zip(props_per_site, tiles, strict=True)):
        w = data_prep.property_weights(props, usable)
        if tile < 0 or w.sum() == 0:
            site_maps.append(np.zeros(1))
            continue
        key = tuple(np.round(w, 6))
        if key not in maps:
            score = me.log_score(log_sim, w)
            maps[key] = (score, me.top_fraction(score, every_tile, ctx.area, ties="mid"))
        score, rank_map = maps[key]
        ranks[i] = rank_map[tile]
        site_maps.append(rank_map)
        tested = np.where(w > 0, log_sim[tile], np.inf)
        worst_var[i] = int(np.argmin(tested))
        worst_sim[i] = float(np.exp(tested[worst_var[i]]))
        best = score.max()
        top_tie[i], at_top[i] = float(ctx.area[score == best].sum()), bool(score[tile] == best)
    return PropertyRanks(ranks, site_maps, worst_var, worst_sim, top_tie, at_top)


def credibility(ctx: Context, t: int, tol_scale: float = 1.0,
                w1_by_target: Mapping[int, np.ndarray] | None = None) -> Credibility:
    """Credibility check of target t, protocol of config.CREDIBILITY_TEST.

    Each known analog of the target's body is ranked among all tiles on the exact properties its
    sources document; the mean rank is compared with random land tiles. The same sites are also
    ranked for the first target of another body (cross-check).
    """
    crit = config.CREDIBILITY_TEST
    known = ctx.ds.known_analogs
    body = ctx.ds.targets["body"].iloc[t]
    props = data_prep.analog_properties(known, body)
    tiles, dist = me.nearest_tiles(ctx.tree, known["lat"].to_numpy(), known["lon"].to_numpy(),
                                   config.CREDIBILITY_MAX_KM)
    w1s = w1_by_target or {}
    own = property_ranks(ctx, t, props, tiles, tol_scale, w1s.get(t))
    test = me.credibility_test_maps(own.ranks, own.maps, crit["top_frac"], int(crit["n_draws"]), crit["alpha"],
                                    int(crit["min_sites"]), tile_weights=ctx.area)
    others = [i for i, b in enumerate(ctx.ds.targets["body"]) if b != body]
    control = (property_ranks(ctx, others[0], props, tiles, tol_scale, w1s.get(others[0])).ranks
               if others else None)
    self_sim = me.self_similarity(ctx.ds.target_quantiles[t], tolerances(ctx, tol_scale))
    measured = np.isfinite(self_sim)
    return Credibility(test, props, dist, own.ranks, control, others[0] if others else None,
                       bool(np.allclose(self_sim[measured], 1.0)), tiles, own.worst_var, own.worst_sim,
                       own.top_tie, own.at_top)


def credibility_all(ctx: Context, tol_scale: float = 1.0,
                    w1_by_target: Mapping[int, np.ndarray] | None = None) -> tuple[list[Credibility], np.ndarray]:
    """Credibility check of every target and its p-value after a Holm correction for testing them all
   . Inconclusive checks (too few sites) are left out of the correction (NaN)."""
    creds = [credibility(ctx, t, tol_scale, w1_by_target) for t in range(len(ctx.ds.targets))]
    p = np.array([c.test.p_value if c.test.conclusive else np.nan for c in creds])
    return creds, me.holm_adjust(p)


@dataclass(frozen=True)
class Atlas:
    percentile: np.ndarray    # (n, T) share of random Earth pairs less similar than the tile is to each target
    top_frac: np.ndarray      # (n, T) share of Earth's land scoring at least as well for each target (0.01 = top 1 %)
    best: np.ndarray          # (n,) target each tile resembles most (highest percentile)
    weights: np.ndarray       # (T, K) variable weights used for each target
    top_overlap: np.ndarray   # (T, T) overlap of the best 1 % of Earth's land of two targets (intersection / union)


def atlas(ctx: Context, dim_weights: Mapping[str, float], tol_scale: float = 1.0,
          w1_by_target: Mapping[int, np.ndarray] | None = None) -> Atlas:
    """Every Earth tile against every target, on an absolute scale (share of Earth pairs).

    Raw scores are not comparable between targets (each one is scored on the variables Earth
    can match for it), so the comparison uses the share of random Earth-Earth pairs that are
    less similar than the tile is to the target, computed with that target's own weights.
    w1_by_target: optional precomputed distances per target index.
    """
    sigma = tolerances(ctx, tol_scale)
    n, n_targets = ctx.ds.n_tiles, len(ctx.ds.targets)
    percentile, top, scores = np.empty((n, n_targets)), np.empty((n, n_targets)), np.empty((n, n_targets))
    weights = np.empty((n_targets, len(config.VARIABLE_NAMES)))
    every_tile = np.arange(n)
    for t in range(n_targets):
        q_target = ctx.ds.target_quantiles[t]
        w1 = (w1_by_target or {}).get(t)
        w1 = me.wasserstein1(ctx.ds.quantiles, q_target) if w1 is None else w1
        w = me.variable_weights_from_dimensions(dim_weights, config.VAR_DIMENSIONS,
                                                scoring_mask(q_target, target_reachability(ctx, w1)))
        score = me.log_score(me.log_similarity(w1, sigma), w)
        percentile[:, t] = me.earth_percentile(score, ctx.pair_gaps, sigma, w)
        top[:, t] = me.top_fraction(score, every_tile, ctx.area)
        weights[t], scores[:, t] = w, score
    return Atlas(percentile, top, percentile.argmax(axis=1), weights,
                 top_land_overlap(top <= config.ROBUSTNESS_TOP_FRAC, ctx.area))


def top_land_overlap(in_top: np.ndarray, area: np.ndarray) -> np.ndarray:
    """(T, T) overlap of the best land of every pair of targets: land area in both sets over land area in either.

    in_top: (n, T) bool, tiles in the best share of Earth's land for each target; area: (n,) land area of the
    tiles. 1 = the same places, 0 = no place in common. A rank correlation over all tiles would not do: every
    barren target ranks forests last, so all Mars targets correlate above 0.9 while their best sites differ.
    """
    a = np.asarray(area, dtype=np.float64)[:, None] * np.asarray(in_top, dtype=np.float64)
    both = a.T @ np.asarray(in_top, dtype=np.float64)
    each = a.sum(axis=0)
    either = each[:, None] + each[None, :] - both
    return np.divide(both, either, out=np.zeros_like(both), where=either > 0)


def twin_targets(top_overlap: np.ndarray,
                 threshold: float = config.TWIN_TARGET_OVERLAP) -> list[tuple[int, int, float]]:
    """Pairs of targets (i < j, overlap) whose best land on Earth is mostly the same places."""
    overlap = np.asarray(top_overlap)
    return [(i, j, float(overlap[i, j])) for i in range(overlap.shape[0]) for j in range(i + 1, overlap.shape[0])
            if overlap[i, j] >= threshold]


SUN_IDX = config.VARIABLE_NAMES.index("sun_elevation_deg")


def lighting_calendar(ctx: Context, t: int, tile: int, tol_scale: float = 1.0) -> np.ndarray:
    """(12,) similarity, month by month, between the Sun's elevation at an Earth site and at target t over its
    year: when to go to find the target's lighting geometry. NaN for a month
    of polar night or when the target has no such profile. Lighting only: the climate is not monthly here.
    """
    target = ctx.ds.target_quantiles[t][SUN_IDX]
    if not np.isfinite(target).all():
        return np.full(12, np.nan)
    monthly = data_prep.earth_monthly_sun_elevation(float(ctx.ds.tile_lat[tile]), ctx.ds.levels)
    gap = np.abs(monthly - target[None, :]).mean(axis=1).astype(np.float32)
    sigma = np.float32(tolerances(ctx, tol_scale)[SUN_IDX])
    return np.exp(me.log_similarity(gap, sigma)).astype(np.float64)


TOPO_IDX: tuple[int, ...] = tuple(i for i, d in enumerate(config.VAR_DIMENSIONS) if d == "Topography")


@dataclass(frozen=True)
class ZoomResult:
    similarity: np.ndarray    # (S,) topographic similarity of each sub-area to the target (NaN = not enough land)
    log_sim: np.ndarray       # (S, T) per topographic variable
    tile_similarity: float    # the same quantity for the whole tile
    order: np.ndarray         # sub-areas with enough land, best first
    weights: np.ndarray       # (T,) weights of the topographic variables (sum = 1)


def zoom_similarity(ctx: Context, t: int, sub_quantiles: np.ndarray, weights: np.ndarray,
                    tile_log_sim: np.ndarray, tol_scale: float = 1.0) -> ZoomResult | None:
    """Topographic similarity to target `t` of the ~11 km sub-areas of a tile.

    sub_quantiles: (S, T, L) quantiles of the topographic variables per sub-area. The target
    profile, the tolerances and the weights are those of the screening, restricted to the
    topographic variables (the only ones known below the tile scale); with no weight on them,
    they weigh the same. tile_log_sim: (K,) log-similarities of the whole tile, for comparison.
    None when the target has no topography.
    """
    idx = np.array(TOPO_IDX)
    q_target = ctx.ds.target_quantiles[t][idx]
    usable = np.isfinite(q_target).all(axis=1)
    if not usable.any():
        return None
    w = np.asarray(weights, dtype=np.float64)[idx] * usable
    w = w / w.sum() if w.sum() > 0 else usable / usable.sum()
    log_sim = me.log_similarity(me.wasserstein1(sub_quantiles, q_target), tolerances(ctx, tol_scale)[idx])
    sim = np.exp(me.log_score(log_sim, w))
    valid = np.flatnonzero(np.isfinite(sim))
    order = valid[np.argsort(-sim[valid], kind="stable")]
    tile_sim = float(np.exp(me.log_score(np.asarray(tile_log_sim)[idx][None, :], w))[0])
    return ZoomResult(sim, log_sim, tile_sim, order, w)


def complementary_sets(ctx: Context, w1: np.ndarray, weights: np.ndarray, tol_scale: float = 1.0,
                       radius_km: float = config.DEFAULT_RADIUS_KM, enforce_min: bool = False, k_max: int = 3):
    """Candidate pool and the best complementary sets for k = 1..k_max.

    Returns (pool tile indices, similarities (M, K) of the pool, list of EnsembleResult).
    """
    w = np.asarray(weights)
    log_sim = me.log_similarity(w1, tolerances(ctx, tol_scale))
    pool = me.candidate_pool(log_sim, w, config.VAR_DIMENSIONS, ctx.xyz, ctx.tree, radius_km,
                             config.POOL_OVERALL, config.POOL_PER_DIMENSION)
    sims = np.exp(log_sim[pool])
    tau = np.where(w > 0, config.COVERED_THRESHOLD, np.nan) if enforce_min else None
    curve = me.ensemble_curve(sims, w, k_max, tau=tau, covered_threshold=config.COVERED_THRESHOLD)
    return pool, sims, curve
