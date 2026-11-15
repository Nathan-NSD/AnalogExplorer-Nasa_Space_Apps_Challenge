"""Tests of the engine orchestration (synthetic dataset, no network)."""
import numpy as np
import pytest

from src import analysis, config
from src import math_engine as me
from tests import synthetic


@pytest.fixture(scope="module")
def ctx():
    return analysis.build_context(synthetic.build_dataset(n_clusters=30, tiles_per_cluster=20, seed=1))


def test_context_shapes(ctx):
    n = ctx.ds.n_tiles
    assert ctx.xyz.shape == (n, 3) and ctx.novelty.shape == (n,) and ctx.understudied.shape == (n,)
    assert ctx.pair_gaps.shape[1] == len(config.VARIABLE_NAMES)


def test_mission_weights_follow_the_profile():
    w = analysis.mission_dimension_weights("Rover Mobility")
    assert w["Topography"] == config.LEVEL_VALUES["strong"] and w["Water / Ice"] == 0.0


def test_scoring_mask_drops_unmeasured_and_unreachable():
    k = len(config.VARIABLE_NAMES)
    q = np.tile(config.QUANTILE_LEVELS, (k, 1)).astype(np.float32)
    q[0] = np.nan                                             # not measured
    reachable = np.ones(k, dtype=bool)
    reachable[1] = False                                      # out of Earth's range
    reach = me.Reachability(np.zeros(k), np.zeros(k, int), np.zeros(k), np.zeros(k), reachable)
    mask = analysis.scoring_mask(q, reach)
    assert not mask[0] and not mask[1] and mask[2:].all()
    nothing = me.Reachability(np.zeros(k), np.zeros(k, int), np.zeros(k), np.zeros(k), np.zeros(k, dtype=bool))
    assert analysis.scoring_mask(q, nothing).tolist() == me.available_variables(q).tolist()   # fallback: measured


def test_weights_fall_back_when_no_weighted_dimension_can_be_scored():
    """Expert weights on Thermal alone, with both thermal variables out of Earth's range (a Moon target): the
    engine uses equal weights on the usable variables, and the app must warn (bug found on 15 Nov 2026)."""
    mask = np.array([d != "Thermal" for d in config.VAR_DIMENSIONS])
    thermal_only = {d: (1.0 if d == "Thermal" else 0.0) for d in config.DIMENSIONS}
    assert analysis.weights_fall_back(thermal_only, mask)
    w = me.variable_weights_from_dimensions(thermal_only, config.VAR_DIMENSIONS, mask)
    assert np.allclose(w[mask], 1 / mask.sum()) and not w[~mask].any()       # what the warning describes
    assert not analysis.weights_fall_back({**thermal_only, "Topography": 0.2}, mask)
    assert not analysis.weights_fall_back(thermal_only, np.ones(len(mask), dtype=bool))


def test_analyse_target_matches_the_engine_primitives(ctx):
    dim_w = analysis.mission_dimension_weights("Rover Mobility")
    a = analysis.analyse_target(ctx, 0, dim_w)
    w1 = me.wasserstein1(ctx.ds.quantiles, ctx.ds.target_quantiles[0])
    assert np.array_equal(a.w1, w1)
    assert np.isclose(a.weights.sum(), 1.0) and (a.weights[~analysis.scoring_mask(
        ctx.ds.target_quantiles[0], a.reach)] == 0).all()
    expected = me.rank_sites(w1, ctx.ds.sigma_default, a.weights, ctx.novelty, 0.0, ctx.xyz, ctx.tree,
                             config.DEFAULT_RADIUS_KM, config.POOL_OVERALL)
    assert np.array_equal(a.ranking.site_idx, expected.site_idx)
    cached = analysis.analyse_target(ctx, 0, dim_w, w1=a.w1, reach=a.reach)   # same result with cached inputs
    assert np.array_equal(cached.ranking.site_idx, a.ranking.site_idx)


def test_best_understudied_is_the_first_flagged_site(ctx):
    sites = np.arange(10)
    flagged = np.flatnonzero(ctx.understudied[sites])
    assert analysis.best_understudied(ctx, sites) == (int(flagged[0]) if flagged.size else None)
    assert analysis.best_understudied(ctx, np.array([], dtype=int)) is None


def test_complementary_sets_grow_with_k(ctx):
    a = analysis.analyse_target(ctx, 0, analysis.mission_dimension_weights("Habitat / Construction"))
    pool, sims, curve = analysis.complementary_sets(ctx, a.w1, a.weights)
    assert sims.shape == (len(pool), len(config.VARIABLE_NAMES)) and len(curve) == 3
    coverage = [r.coverage for r in curve]
    assert coverage == sorted(coverage) and coverage[-1] <= curve[0].ceiling + 1e-9


def test_zoom_similarity_finds_the_sub_area_matching_the_target(ctx):
    topo = list(analysis.TOPO_IDX)
    target = ctx.ds.target_quantiles[0][topo]
    sub = np.stack([target, target * 3.0, np.full_like(target, np.nan)])      # perfect, poor, no land
    weights = np.zeros(len(config.VARIABLE_NAMES))
    weights[topo] = 0.5
    res = analysis.zoom_similarity(ctx, 0, sub, weights, np.zeros(len(config.VARIABLE_NAMES)))
    assert res.order.tolist() == [0, 1] and res.similarity[0] == pytest.approx(1.0)
    assert res.similarity[1] < 1.0 and np.isnan(res.similarity[2])
    assert res.tile_similarity == pytest.approx(1.0) and res.weights.sum() == pytest.approx(1.0)
    no_weight = analysis.zoom_similarity(ctx, 0, sub, np.zeros(len(config.VARIABLE_NAMES)),
                                         np.zeros(len(config.VARIABLE_NAMES)))
    assert no_weight.weights.tolist() == [0.5, 0.5]                          # falls back to equal weights


def test_zoom_similarity_needs_target_topography(ctx):
    saved = ctx.ds.target_quantiles[0].copy()
    try:
        ctx.ds.target_quantiles[0][list(analysis.TOPO_IDX)] = np.nan
        sub = np.zeros((2, len(analysis.TOPO_IDX), config.QUANTILE_LEVELS.size), dtype=np.float32)
        assert analysis.zoom_similarity(ctx, 0, sub, np.ones(len(config.VARIABLE_NAMES)),
                                        np.zeros(len(config.VARIABLE_NAMES))) is None
    finally:
        ctx.ds.target_quantiles[0] = saved


def test_atlas_ranks_every_tile_against_every_target(ctx):
    dim_w = analysis.mission_dimension_weights("Rover Mobility")
    atlas = analysis.atlas(ctx, dim_w)
    n, n_targets = ctx.ds.n_tiles, len(ctx.ds.targets)
    assert atlas.percentile.shape == atlas.top_frac.shape == (n, n_targets) and atlas.best.shape == (n,)
    assert (atlas.percentile >= 0).all() and (atlas.percentile <= 1).all()
    assert (atlas.top_frac > 0).all() and (atlas.top_frac <= 1).all()
    assert np.array_equal(atlas.best, atlas.percentile.argmax(axis=1))
    a = analysis.analyse_target(ctx, 2, dim_w)                         # same numbers as the per-target analysis
    assert np.allclose(atlas.weights[2], a.weights)
    expected = me.earth_percentile(a.ranking.log_score, ctx.pair_gaps, ctx.ds.sigma_default, a.weights)
    assert np.allclose(atlas.percentile[:, 2], expected)
    best_tile = int(np.argmax(a.ranking.log_score))
    assert atlas.top_frac[best_tile, 2] == pytest.approx(ctx.area[best_tile])     # a share of the land area
    assert atlas.top_overlap.shape == (n_targets, n_targets) and np.allclose(np.diag(atlas.top_overlap), 1.0)
    assert np.allclose(atlas.top_overlap, atlas.top_overlap.T) and (atlas.top_overlap <= 1 + 1e-9).all()
    cached = analysis.atlas(ctx, dim_w, w1_by_target={2: a.w1})         # precomputed distances give the same atlas
    assert np.array_equal(cached.percentile, atlas.percentile)


def test_credibility_reports_what_explains_each_rank(ctx, monkeypatch):
    """Regression (14 Nov 2026 audit): every ranked known analog comes with its tile, its worst tested variable
    and the size of the tie at the best score, so that the app can explain a low rank."""
    monkeypatch.setattr(config, "CREDIBILITY_MAX_KM", 25_000.0)   # sparse synthetic set: always find a tile
    cred = analysis.credibility(ctx, 0)
    ranked = np.isfinite(cred.ranks)
    assert ranked.any() and (cred.tiles[ranked] >= 0).all()
    assert (cred.worst_var[ranked] >= 0).all() and (cred.worst_var[~ranked] == -1).all()
    assert ((cred.worst_sim[ranked] > 0) & (cred.worst_sim[ranked] <= 1)).all()
    assert ((cred.top_tie[ranked] > 0) & (cred.top_tie[ranked] <= 1)).all()
    assert (cred.ranks[cred.at_top] <= cred.top_tie[cred.at_top] + 1e-9).all()      # a tied site ranks 'top <tie>'
    assert 0 < cred.test.null_mean < 1


# ------------------------------------------------------------ land-area weights and ties
def test_context_weighs_tiles_by_land_area(ctx):
    expected = np.cos(np.radians(ctx.ds.tile_lat)) * ctx.ds.valid_frac
    assert ctx.area.sum() == pytest.approx(1.0) and np.allclose(ctx.area, expected / expected.sum())
    score = np.arange(ctx.ds.n_tiles, dtype=float)                       # tile n-1 is the best
    order = np.argsort(-score)
    assert np.allclose(analysis.top_share(ctx, score, order), np.cumsum(ctx.area[order]))


def test_twin_targets_share_most_of_their_best_land():
    """Overlap = land in both best sets / land in either; a rank correlation over all tiles flagged every pair
    of Mars targets (0.90 to 1.00) although their best sites differ."""
    in_top = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
    area = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    overlap = analysis.top_land_overlap(in_top, area)
    assert overlap[0, 1] == pytest.approx(0.6 / 0.8) and overlap[0, 2] == 0 and np.allclose(np.diag(overlap), 1)
    assert analysis.twin_targets(overlap, 0.5) == [(0, 1, pytest.approx(0.75))]
    assert analysis.twin_targets(overlap, 0.9) == []


def test_lighting_calendar_prefers_the_months_with_the_targets_sun(ctx):
    """A target with a grazing Sun (2 deg) is best matched, at 70 N, near the start and end of the polar day,
    not at midsummer; a month of polar night has no value."""
    ds = ctx.ds
    tile = int(np.argmin(np.abs(ds.tile_lat - 70.0)))
    lat = float(ds.tile_lat[tile])
    saved = ds.target_quantiles[0, analysis.SUN_IDX].copy()
    try:
        ds.target_quantiles[0, analysis.SUN_IDX] = 2.0
        cal = analysis.lighting_calendar(ctx, 0, tile)
    finally:
        ds.target_quantiles[0, analysis.SUN_IDX] = saved
    assert cal.shape == (12,)
    if lat > 68:                                                         # polar night in December
        assert np.isnan(cal[11])
    lit = np.isfinite(cal)
    assert ((cal[lit] > 0) & (cal[lit] <= 1)).all()
    assert cal[5] < np.nanmax(cal)                                       # June (high Sun) is not the best month
