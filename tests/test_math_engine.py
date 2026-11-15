"""Sanity tests of the numerical core."""
from itertools import combinations

import numpy as np
import pytest

from src import config
from src import math_engine as me


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _sorted_quantiles(rng, n, k, n_levels=19):
    return np.sort(rng.normal(size=(n, k, n_levels)), axis=2).astype(np.float32)


def test_self_similarity_is_one(rng):
    q = _sorted_quantiles(rng, 1, 4)[0]
    assert np.allclose(me.self_similarity(q, np.ones(4)), 1.0)


def test_wasserstein_is_mean_quantile_gap(rng):
    q = _sorted_quantiles(rng, 5, 3)
    target = q[0] + 2.0
    assert np.allclose(me.wasserstein1(q[:1], target), 2.0)


def test_normalize_weights():
    assert np.allclose(me.normalize_weights([1, 1, 2]), [0.25, 0.25, 0.5])
    with pytest.raises(ValueError):
        me.normalize_weights([0, 0])
    with pytest.raises(ValueError):
        me.normalize_weights([-1, 2])


def test_dimension_weights_spread_over_variables():
    w = me.variable_weights_from_dimensions({"A": 1.0, "B": 1.0}, ["A", "A", "B"])
    assert np.allclose(w, [0.25, 0.25, 0.5])


def test_reachability_flags_a_variable_outside_earth(rng):
    q = _sorted_quantiles(rng, 300, 2)
    target = np.stack([q[0, 0], q[0, 1] + 50.0])        # variable 1 far beyond any Earth tile
    w1 = me.wasserstein1(q, target)
    gaps = me.earth_pair_gaps(q, n_pairs=2000)
    reach = me.reachability(w1, gaps, np.median(gaps, axis=0))
    assert reach.reachable.tolist() == [True, False]
    assert reach.best_tile[0] == 0 and reach.percentile[1] == 1.0 and reach.gap_sigma[1] > 10


def test_earth_percentile_is_monotone(rng):
    q = _sorted_quantiles(rng, 200, 3)
    gaps = me.earth_pair_gaps(q, n_pairs=2000)
    sigma, w = np.median(gaps, axis=0), np.full(3, 1 / 3)
    pct = me.earth_percentile(np.array([-1e9, -0.5, 0.0]), gaps, sigma, w)
    assert pct[0] == 0.0 and pct[2] == 1.0 and 0.0 < pct[1] < 1.0


def test_structural_draws(rng):
    w = np.array([0.5, 0.3, 0.2, 0.0])
    active, draws = me._dirichlet_draws(w, (5.0, 50.0), 600, rng, drop_prob=0.5)
    assert active.tolist() == [0, 1, 2] and np.allclose(draws.sum(axis=1), 1.0)
    dropped = (draws < 1e-12).any(axis=1).mean()
    assert 0.35 < dropped < 0.65                                    # about half of the draws leave a variable out


def test_understudied_excludes_famous_regions_and_known_sites():
    lat, lon = np.array([-75.0, 10.0, 10.0, -24.0]), np.array([0.0, 20.0, 20.0, -69.0])
    dist = np.array([2000.0, 2000.0, 100.0, 2000.0])
    regions = {"Antarctica": (-90, -60, -180, 180), "Atacama": (-30, -18, -72, -66)}
    assert me.understudied(lat, lon, dist, 500.0, regions).tolist() == [False, True, False, False]


def test_credibility_test_maps_uses_each_site_map(rng):
    good, bad = np.linspace(0, 1, 1000), np.linspace(1, 0, 1000)      # two rank maps over the same tiles
    res = me.credibility_test_maps(np.array([0.01] * 6), [good] * 6, n_draws=2000)
    assert res.passed and res.n_sites == 6
    res = me.credibility_test_maps(np.array([0.01, np.nan]), [good, bad], n_draws=2000)
    assert res.n_sites == 1 and not res.conclusive


def test_missing_variable_weight_goes_to_its_dimension():
    dims = ["A", "A", "B"]
    w = me.variable_weights_from_dimensions({"A": 1.0, "B": 1.0}, dims, np.array([True, False, True]))
    assert np.allclose(w, [0.5, 0.0, 0.5])
    w = me.variable_weights_from_dimensions({"A": 1.0, "B": 0.0}, dims, np.array([False, False, True]))
    assert np.allclose(w, [0.0, 0.0, 1.0])  # nothing positive left: equal weights on what is measured


def test_not_evaluated_variable_is_ignored(rng):
    q = _sorted_quantiles(rng, 6, 3)
    target = q[0].copy()
    target[2] = np.nan
    avail = me.available_variables(target)
    assert avail.tolist() == [True, True, False]
    log_sim = me.log_similarity(me.wasserstein1(q, target), np.ones(3))
    w = me.variable_weights_from_dimensions({"A": 1.0, "B": 1.0}, ["A", "A", "B"], avail)
    score = me.log_score(log_sim, w)
    assert np.isfinite(score).all() and score[0] == pytest.approx(0.0)
    dim = me.dimension_log_similarity(log_sim, ["A", "A", "B"], ["A", "B"])
    assert np.isfinite(dim[:, 0]).all() and np.isnan(dim[:, 1]).all()
    res = me.best_complementary_set(np.exp(log_sim), w, 2)
    assert np.isfinite(res.coverage) and not res.uncovered[2] and np.isnan(res.best_per_variable[2])


def test_non_max_suppression_respects_radius(rng):
    lat, lon = rng.uniform(-60, 60, 400), rng.uniform(-180, 180, 400)
    xyz = me.unit_xyz(lat, lon)
    picked = me.non_max_suppression(rng.random(400), xyz, 800.0, 50)
    d = me.haversine_km(lat[picked][:, None], lon[picked][:, None], lat[picked][None], lon[picked][None])
    assert (d[np.triu_indices(len(picked), 1)] > 800.0).all()


def test_pareto_front():
    a = np.array([1.0, 2.0, 3.0, 2.5])
    b = np.array([3.0, 2.0, 1.0, 0.5])
    assert me.pareto_front(a, b).tolist() == [True, True, True, False]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_exhaustive_ensemble_matches_brute_force(rng, k):
    S = rng.random((12, 5)).astype(np.float32)
    w = me.normalize_weights(rng.random(5))
    res = me.best_complementary_set(S, w, k, method="exhaustive")
    brute = max(me.coverage(S, w, c) for c in combinations(range(12), k))
    assert res.coverage == pytest.approx(brute, rel=1e-5)
    assert res.best_single <= res.coverage + 1e-6 <= res.ceiling + 2e-6


def test_greedy_reaches_submodular_bound(rng):
    S = rng.random((40, 6)).astype(np.float32)
    w = me.normalize_weights(rng.random(6))
    greedy = me.best_complementary_set(S, w, 3, method="greedy").coverage
    exact = me.best_complementary_set(S, w, 3, method="exhaustive").coverage
    assert greedy >= (1 - 1 / np.e) * exact


def test_score_interval_brackets_nominal_score(rng):
    w1 = rng.random((50, 4)).astype(np.float32)
    sigma = np.ones(4, dtype=np.float32)
    w = me.normalize_weights([1, 1, 1, 1])
    idx = np.arange(10)
    nominal = np.exp(me.log_score(me.log_similarity(w1, sigma), w))[idx]
    lo, hi = me.score_interval(w1, sigma, w, idx, n_draws=400).T
    assert ((lo <= nominal + 1e-6) & (nominal <= hi + 1e-6)).all()


def test_robustness_is_a_probability(rng):
    w1 = rng.random((200, 3)).astype(np.float32)
    r = me.robustness(w1, np.ones(3), me.normalize_weights([1, 2, 3]), n_draws=50, top_frac=0.05)
    assert r.shape == (200,) and (r >= 0).all() and (r <= 1).all()
    assert r.sum() == pytest.approx(50 * 10 / 50)  # exactly 10 tiles in the top 5 % per draw


def test_credibility_test_detects_good_and_random_rankings(rng):
    score = rng.normal(size=5000)
    top = me.top_fraction(score, np.argsort(-score)[:8])            # 8 sites among the very best tiles
    good = me.credibility_test(score, top, n_draws=2000)
    assert good.passed and good.p_value < 0.01 and good.n_in_top == 8 and good.binomial_p < 1e-6
    rand = me.credibility_test(score, me.top_fraction(score, rng.integers(0, 5000, 8)), n_draws=2000)
    assert rand.p_value > 0.01 or not rand.passed
    few = me.credibility_test(score, top[:3], n_draws=2000)
    assert not few.conclusive and not few.passed                    # 3 sites: no power, inconclusive
    assert me.credibility_test(score, np.array([np.nan])).n_sites == 0


# ------------------------------------------------------------ Huber penalty
def test_log_similarity_is_quadratic_then_linear():
    sigma = np.array([2.0], dtype=np.float32)
    w1 = np.array([[0.0], [2.0], [5.0], [8.0], [20.0]], dtype=np.float32)     # z = 0, 1, 2.5, 4, 10
    huber = me.log_similarity(w1, sigma, delta=2.5)[:, 0]
    gauss = me.log_similarity(w1, sigma, delta=None)[:, 0]
    assert np.allclose(gauss, [0.0, -0.5, -3.125, -8.0, -50.0])
    assert np.allclose(huber[:3], gauss[:3])                                  # identical up to delta
    assert np.allclose(huber[3:], [-(2.5 * 4 - 3.125), -(2.5 * 10 - 3.125)])  # linear beyond
    assert (np.diff(huber) < 0).all() and (huber >= gauss).all()              # still decreasing, never harsher
    slope_before = (me.log_similarity(np.float32([[4.99]]), sigma, delta=2.5)
                    - me.log_similarity(np.float32([[5.0]]), sigma, delta=2.5))[0, 0] / 0.005
    slope_after = (me.log_similarity(np.float32([[5.0]]), sigma, delta=2.5)
                   - me.log_similarity(np.float32([[5.01]]), sigma, delta=2.5))[0, 0] / 0.005
    assert slope_before == pytest.approx(slope_after, rel=0.02)               # no kink at the junction


def test_log_similarity_keeps_missing_values_and_dtype():
    w1 = np.array([[np.nan, 1.0, 30.0]], dtype=np.float32)
    out = me.log_similarity(w1, np.ones(3, dtype=np.float32))
    assert out.dtype == np.float32 and np.isnan(out[0, 0]) and np.isfinite(out[0, 1:]).all()
    assert me.log_similarity(w1, np.ones(3, dtype=np.float32)).shape == w1.shape


def test_the_default_penalty_is_the_configured_one():
    w1 = np.array([[10.0]], dtype=np.float32)
    assert np.allclose(me.log_similarity(w1, np.float32([1.0])),
                       me.log_similarity(w1, np.float32([1.0]), delta=config.HUBER_DELTA))


def test_robustness_and_interval_use_the_same_penalty():
    rng = np.random.default_rng(0)
    w1 = rng.gamma(2.0, 2.0, size=(200, 3)).astype(np.float32)                # many gaps beyond 2.5 sigma
    sigma, w = np.ones(3, dtype=np.float32), np.array([0.5, 0.3, 0.2])
    idx = np.arange(5)
    interval = me.score_interval(w1, sigma, w, idx, alpha=1e9, n_draws=20)    # huge alpha: weights barely move
    exact = np.exp(me.log_score(me.log_similarity(w1[idx], sigma), w))
    assert np.allclose(interval[:, 0], exact, rtol=1e-2) and np.allclose(interval[:, 1], exact, rtol=1e-2)
    best = int(np.argmax(me.log_score(me.log_similarity(w1, sigma), w)))
    assert me.robustness(w1, sigma, w, alpha=1e9, n_draws=20, top_frac=0.01)[best] == 1.0


def test_default_tolerance_has_no_jump_when_pairs_are_identical(rng):
    """sigma = P(differ) x median gap of the differing pairs: the median for a continuous variable, and no jump
    when the share of identical pairs crosses one half (bare ground fell from 0.18 to 0.004)."""
    q = _sorted_quantiles(rng, 400, 2)
    sigmas = []
    for n_zero in (270, 290, 300):                               # 45 %, 52 % and 56 % of identical pairs
        q2 = q.copy()
        q2[:n_zero, 1] = 0.0                                     # these tiles have none of the second variable
        gaps = me.earth_pair_gaps(q2)
        differ = gaps[:, 1] > 0
        sigma = me.default_tolerances(q2)
        assert sigma[1] == pytest.approx(differ.mean() * np.median(gaps[differ, 1]), rel=1e-5)
        assert sigma[0] == pytest.approx(np.median(gaps[:, 0]), rel=1e-5)   # continuous variable: the median
        sigmas.append(float(sigma[1]))
    assert max(sigmas) / min(sigmas) < 1.5                       # the former rule changed by a factor of 40


def test_closer_than_share_ignores_identical_pairs():
    # variable 0 continuous; variable 1 bounded: half of the pairs are identical (gap 0)
    gaps = np.column_stack([np.linspace(0.01, 1.0, 100), np.r_[np.zeros(50), np.linspace(0.02, 1.0, 50)]])
    perfect = me.closer_than_share(np.array([0.0, 0.0]), gaps)
    assert perfect.tolist() == [1.0, 1.0]                       # a perfect match is never "weak"
    mid = me.closer_than_share(np.array([0.5, 0.25]), gaps)
    assert mid[0] == pytest.approx((gaps[:, 0] > 0.5).mean())   # continuous variable: unchanged
    # among the pairs that differ, the gap being read divided by their share (0.25 / 0.5 = 0.5)
    assert mid[1] == pytest.approx((gaps[:, 1] > 0.5).mean() / 0.5)
    worst = me.closer_than_share(np.array([1.0, 1.0]), gaps)
    assert worst.tolist() == [0.0, 0.0]
    assert np.isnan(me.closer_than_share(np.array([np.nan, 0.1]), gaps)[0])


def test_holm_adjust_matches_the_step_down_rule():
    p = np.array([0.0011, 0.0030, 0.0406, 0.0427, 0.0412, 0.9485, np.nan])
    adj = me.holm_adjust(p)
    assert adj[0] == pytest.approx(6 * 0.0011) and adj[1] == pytest.approx(5 * 0.0030)
    assert adj[2] == pytest.approx(4 * 0.0406)                       # third smallest, times 4
    assert adj[4] == pytest.approx(4 * 0.0406)                       # monotone: never below the previous one
    assert adj[3] == pytest.approx(4 * 0.0406) and adj[5] == pytest.approx(0.9485) and np.isnan(adj[6])
    assert ((adj[:6] < 0.05) == np.array([True, True, False, False, False, False])).all()


# ------------------------------------------------------------ regressions of the 14 Nov 2026 audit
def test_relative_score_interval_never_exceeds_one(rng):
    """The relative score is 1 for the best tile of a draw: its range under weight perturbation stays in (0, 1].
    Dividing absolute ranges by the unperturbed best score gave ranges up to 2.9."""
    w1 = (3 * rng.random((300, 4))).astype(np.float32)
    sigma = np.ones(4, dtype=np.float32)
    w = me.normalize_weights([1, 2, 1, 4])
    log_score = me.log_score(me.log_similarity(w1, sigma), w)
    idx = np.argsort(-log_score)[:10]
    settings = dict(alpha=config.DIRICHLET_ALPHAS, n_draws=200, drop_prob=config.ROBUSTNESS_DROP_PROB)
    lo, hi = me.score_interval(w1, sigma, w, idx, relative=True, **settings).T
    assert (lo > 0).all() and (hi <= 1 + 1e-6).all() and (lo <= hi).all()
    nominal = np.exp(log_score[idx] - log_score.max())
    assert lo[0] <= nominal[0] <= hi[0] + 1e-6          # the best tile stays near 1
    old = me.score_interval(w1, sigma, w, idx, **settings)[:, 1] / np.exp(log_score.max())
    assert old.max() > 1                                # the former normalisation did exceed 1 on this case


def test_tied_tiles_share_their_average_rank(rng):
    """With 45 % of the tiles tied at the best score, counting ties as ahead ranked a perfect match 'top 45 %'
    and random tiles 'top 61 %' on average; the average rank of the tie gives 'top 22.5 %' and 50 %."""
    score = np.where(rng.random(4000) < 0.45, 0.0, -rng.random(4000))     # 45 % of the tiles tied for first place
    every = np.arange(score.size)
    worst, mid = me.top_fraction(score, every), me.top_fraction(score, every, ties="mid")
    best = int(np.argmax(score))
    tie = float((score == 0.0).mean())
    assert worst[best] == pytest.approx(tie) and mid[best] == pytest.approx(tie / 2)
    assert worst.mean() > 0.55 and mid.mean() == pytest.approx(0.5, abs=1e-3)
    tied = me.credibility_test(score, mid[:8], n_draws=4000)
    assert tied.null_mean == pytest.approx(0.5, abs=0.02)
    with pytest.raises(ValueError):
        me.top_fraction(score, every, ties="best")


def test_top_fraction_is_a_share_of_the_land_area():
    score = np.array([3.0, 2.0, 1.0, 0.0])
    area = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(me.top_fraction(score, np.arange(4), area), [0.1, 0.3, 0.6, 1.0])
    assert np.allclose(me.top_fraction(score, np.arange(4)), [0.25, 0.5, 0.75, 1.0])      # counts of tiles
    assert np.allclose(me.top_fraction(score, np.arange(4), area, ties="mid"), [0.05, 0.2, 0.45, 0.8])


def test_land_area_weights_shrink_toward_the_poles():
    w = me.land_area_weights(np.array([0.0, 60.0, 60.0]), np.array([1.0, 1.0, 0.5]))
    assert w.sum() == pytest.approx(1.0)
    assert w[1] == pytest.approx(w[0] / 2) and w[2] == pytest.approx(w[1] / 2)


def test_pairs_are_drawn_by_land_area(rng):
    """Tiles without area never enter the reference pairs: the yardstick is two places on land, not two tiles."""
    q = _sorted_quantiles(rng, 200, 2)
    q[100:] += 50.0                                                       # a second, very different half
    weights = np.r_[np.ones(100), np.zeros(100)]
    assert me.earth_pair_gaps(q, 2000, weights=weights).max() < 10
    assert me.earth_pair_gaps(q, 2000).max() > 40                        # uniform draw mixes the two halves
    assert (me.default_tolerances(q, weights=weights) < me.default_tolerances(q)).all()


def test_robustness_by_area_keeps_the_best_share_of_the_land(rng):
    w1 = rng.random((200, 3)).astype(np.float32)
    area = rng.random(200)
    weights = me.normalize_weights([1, 2, 3])
    r = me.robustness(w1, np.ones(3), weights, n_draws=40, top_frac=0.05, tile_weights=area)
    assert r.shape == (200,) and (r >= 0).all() and (r <= 1).all() and r.max() > 0
    assert float(area @ r) / area.sum() == pytest.approx(0.05, abs=0.02)   # about 5 % of the land per draw
    equal = me.robustness(w1, np.ones(3), weights, n_draws=40, top_frac=0.05, tile_weights=np.ones(200))
    assert np.allclose(equal, me.robustness(w1, np.ones(3), weights, n_draws=40, top_frac=0.05))


def test_credibility_null_draws_places_by_area(rng):
    ranks = np.linspace(0.005, 1.0, 200)
    area = np.r_[np.ones(100), np.zeros(100)]                             # only the best half of the tiles holds land
    res = me.credibility_test_maps(np.full(6, 0.2), [ranks] * 6, n_draws=2000, tile_weights=area)
    assert res.null_mean == pytest.approx(ranks[:100].mean(), abs=0.02)


def test_similarity_at_closer_share_inverts_the_share(rng):
    """A site exactly at the returned similarity is closer than `share` of the Earth pairs that differ."""
    gaps = np.abs(rng.normal(size=(5000, 3))).astype(np.float32)
    gaps[:2500, 2] = 0.0                                                  # half of the pairs identical on variable 2
    sigma = np.array([1.0, 0.5, 1.0], dtype=np.float32)
    thr = me.similarity_at_closer_share(gaps, sigma, 0.75)
    w = sigma * np.sqrt(-2 * np.log(thr))                                 # back to a W1 gap (below the Huber knee)
    assert np.allclose(me.closer_than_share(w, gaps), 0.75, atol=0.01)
    assert not np.allclose(thr, thr[0])                                   # one threshold per variable, not a circle


# ------------------------------------------------------------ one yardstick for score and status
def test_one_tolerance_reads_fifty_percent_for_every_variable(rng):
    """The status is read in the unit of the score: a site one default tolerance away from the target is closer
    than 50 % of the Earth pairs that differ, whether 0 %, 40 % or 60 % of the pairs are identical."""
    q = _sorted_quantiles(rng, 600, 3)
    q[:230, 1] = 0.0                                            # ~15 % of identical pairs on variable 1
    q[:460, 2] = 0.0                                            # ~59 % of identical pairs on variable 2
    gaps = me.earth_pair_gaps(q)
    sigma = me.default_tolerances(q)
    assert (gaps[:, 2] == 0).mean() > 0.5 > (gaps[:, 1] == 0).mean() > 0.1
    at_one = me.closer_than_share(sigma.astype(np.float64), gaps)
    assert np.allclose(at_one, 0.5, atol=0.01)
    # the label follows the similarity: closer in tolerances means a higher share, for every variable alike
    half, double = me.closer_than_share(0.5 * sigma, gaps), me.closer_than_share(2.0 * sigma, gaps)
    assert (half > at_one).all() and (double < at_one).all()
    strong = me.similarity_at_closer_share(gaps, sigma, 0.5)    # the radar threshold is the same reading
    assert np.allclose(strong, np.exp(-0.5), atol=0.01)


def test_a_seasonally_frozen_site_is_not_a_strong_match_for_permafrost():
    """Regression (15 Nov 2026): on a nearly binary variable (ground below 0 C: 35 % of the pairs differ, almost
    all of them by the whole range), a site frozen a quarter of the year read 'closer than 92 % of Earth pairs'
    to a permanently frozen target, hence a strong match, while its similarity was 0.10."""
    rng = np.random.default_rng(1)
    frozen = (rng.random(4000) < 0.2).astype(np.float32)        # 20 % of the places are frozen all year
    q = np.repeat(frozen[:, None, None], 19, axis=2)
    gaps = me.earth_pair_gaps(q)
    sigma = me.default_tolerances(q)
    assert sigma[0] == pytest.approx((gaps[:, 0] > 0).mean(), rel=1e-5)          # P(differ) x median gap of 1
    quarter, most = np.array([0.75]), np.array([0.1])           # gaps to a target frozen all year
    assert me.closer_than_share(quarter, gaps)[0] == 0.0        # weak, like its similarity
    assert np.exp(me.log_similarity(quarter.astype(np.float32), sigma))[0] < 0.2
    assert me.closer_than_share(most, gaps)[0] == 1.0           # frozen 90 % of the year: still a strong match
    assert np.exp(me.log_similarity(most.astype(np.float32), sigma))[0] > 0.9
