"""Tests of the rule-based report."""
import numpy as np
import pandas as pd

from src import config, report
from tests import synthetic


def _log_sim(values):
    return np.log(np.asarray(values, dtype=np.float64))


def test_status_thresholds():
    assert report.status_label(0.9) == "Strong match"
    assert report.status_label(0.6) == "Partial match"
    assert report.status_label(0.1) == "Weak match"
    assert report.status_label(float("nan")) == "Not evaluated"


def test_not_evaluated_dimension_is_reported():
    k = len(config.VARIABLES)
    q = np.tile(config.QUANTILE_LEVELS, (k, 1))
    illum = [i for i, v in enumerate(config.VARIABLES) if v.dimension == "Illumination"]
    log_sim = _log_sim(np.full(k, 0.9))
    log_sim[illum] = np.nan
    target_q = q.copy()
    target_q[illum] = np.nan
    profile = report.profile_table(q, target_q, log_sim, config.QUANTILE_LEVELS)
    assert set(profile.loc[illum, "Status"]) == {"Not evaluated"}
    statements = report.limits_statements(log_sim, profile)
    assert any(s.startswith("Illumination: not evaluated") for s in statements)
    plan = report.site_test_plan(log_sim, {d: 1.0 for d in config.DIMENSIONS}, "Moon")
    assert any("not evaluated" in s for s in plan["Not representative here"])


def test_status_from_earth_reference():
    assert report.status_label(0.1, closer=0.8) == "Strong match"      # similarity ignored when closer is given
    assert report.status_label(0.9, closer=0.6) == "Partial match"
    assert report.status_label(0.9, closer=0.2) == "Weak match"
    assert report.status_label(0.9, closer=float("nan")) == "Not evaluated"
    closer = np.full(len(config.VARIABLES), 0.9)
    summary = report.dimension_summary(_log_sim(np.full(len(config.VARIABLES), 0.1)), closer)
    assert set(summary["Status"]) == {"Strong match"} and (summary[report.CLOSER_COLUMN] == 90.0).all()


def test_candidate_table_shows_relative_scores():
    log_score = np.log(np.array([1e-20, 1e-18, 1e-19]))
    idx = np.array([1, 2])
    df = report.candidate_table(idx, np.zeros(3), np.zeros(3), log_score, np.zeros(3), ["a"], np.zeros(3, int),
                                np.zeros(3), np.array([0.0, 0.5]), earth_pct=np.array([0.9, 0.4]))
    assert np.allclose(df["Relative score"], [1.0, 0.1]) and "Score" not in df.columns
    assert df["Closer than % of Earth pairs"].tolist() == [90.0, 40.0]


def test_challenge_table_covers_the_official_text():
    table = report.challenge_table()
    assert (table["Type"] == "Objective").sum() == 4 and (table["Type"] == "Consideration").sum() == 12
    assert set(table["Status"]) <= set(config.CHALLENGE_STATUSES)
    assert table.notna().all().all() and (table.astype(str).apply(lambda col: col.str.len()) > 0).all().all()
    counts = report.challenge_counts()
    assert sum(counts.values()) == len(table) and list(counts) == list(config.CHALLENGE_STATUSES)


def test_worldview_url_frames_the_site():
    url = report.worldview_url(10.5, -20.5, half_deg=0.5)
    assert url.startswith("https://worldview.earthdata.nasa.gov/") and "v=-21.00,10.00,-20.00,11.00" in url


def test_structural_limits_filtered_by_body():
    df = report.structural_limits_table("Mars")
    assert "On Mars" in df.columns and "Moon" not in df.columns


def test_site_test_plan_follows_mission_and_status():
    k = len(config.VARIABLES)
    sims = np.full(k, 0.95)
    sims[[i for i, v in enumerate(config.VARIABLES) if v.dimension == "Illumination"]] = 0.1
    weights = {"Topography": 1.0, "Thermal": 0.0, "Water / Ice": 0.0, "Illumination": 1.0}
    plan = report.site_test_plan(_log_sim(sims), weights, "Moon")
    assert set(plan["Representative tests"]) == set(config.TEST_ACTIVITIES["Topography"])
    assert any(s.startswith("Illumination") for s in plan["Not representative here"])
    assert len(plan["Never transferable from Earth"]) == len(config.STRUCTURAL_LIMITS)


def test_site_sheet_markdown_contains_sections():
    k = len(config.VARIABLES)
    q = np.tile(config.QUANTILE_LEVELS, (k, 1))
    log_sim = _log_sim(np.full(k, 0.7))
    profile = report.profile_table(q, q, log_sim, config.QUANTILE_LEVELS)
    plan = report.site_test_plan(log_sim, {}, "Mars")
    md = report.site_sheet_markdown("Gale", "Mars", "Rover Mobility", "#1 (0, 0)", {"Score": "0.7"}, profile,
                                    report.limits_statements(log_sim, profile), plan, {"source": "synthetic"})
    for section in ("## Profile by variable", "## What to test here", "Possible with caution"):
        assert section in md


def test_zoom_table_lists_the_best_sub_areas():
    levels = config.QUANTILE_LEVELS
    bounds = np.array([[10.0, 10.1, 20.0, 20.1], [10.1, 10.2, 20.0, 20.1], [10.2, 10.3, 20.0, 20.1]])
    q = np.stack([np.stack([np.full(levels.size, 5.0 * (i + 1)), np.full(levels.size, 40.0)]) for i in range(3)])
    table = report.zoom_table(bounds, q, np.array([0.2, 0.9, np.nan]), np.array([1, 0]), levels)
    assert table["Rank"].tolist() == [1, 2] and table["Lat"].iloc[0] == 10.15
    assert table["Topographic similarity"].tolist() == [0.9, 0.2]
    assert table["Slope (median)"].iloc[0] == "10.0 deg" and table["Size (km)"].iloc[0] == "11 x 11"
    assert table["NASA Worldview"].iloc[0].startswith("https://worldview.earthdata.nasa.gov/")


def test_subsurface_table_orders_by_temperature_gap():
    import pandas as pd

    sites = pd.DataFrame({"name": ["cold", "close", "hot"], "country": ["A", "B", "C"], "lat": [60.0, 30.0, 0.0],
                          "lon": [10.0, 20.0, 30.0], "maat_k": [270.0, 291.5, 300.0], "source": ["s"] * 3})
    table = report.subsurface_table(sites, 290.0, np.array([0.5, np.nan, 0.01]), np.array([True, False, True]))
    assert table["Lava-tube field"].tolist() == ["close", "hot", "cold"]
    assert table[report.SUBSURFACE_GAP].tolist() == [1.5, 10.0, -20.0]
    assert np.isnan(table["Surface match (top % of Earth's land)"].iloc[0]) and table["Under-studied"].tolist() == [
        False, True, True]
    assert "NASA Worldview" in report.subsurface_table(sites, 290.0).columns


def test_reverse_table_puts_the_closest_target_first():
    k = len(config.VARIABLES)
    log_sim = _log_sim(np.full((2, k), 0.9))
    closer = np.full((2, k), 0.6)
    illum = [i for i, v in enumerate(config.VARIABLES) if v.dimension == "Illumination"]
    topo = [i for i, v in enumerate(config.VARIABLES) if v.dimension == "Topography"]
    closer[1, illum], closer[1, topo] = 0.1, 0.95
    log_sim[0, illum] = np.nan                                  # not evaluated for the first target
    closer[0, illum] = np.nan
    table = report.reverse_table(["Far", "Near"], ["Moon", "Mars"], np.array([0.3, 0.9]), np.array([0.5, 0.0002]),
                                 log_sim, closer)
    assert table["Target"].tolist() == ["Near", "Far"] and table[report.CLOSER_COLUMN].tolist() == [90.0, 30.0]
    assert table[report.RANK_COLUMN].tolist() == [0.02, 50.0]    # share of Earth's land, in percent
    assert table["Strongest dimension"].iloc[0] == "Topography (strong match)"
    assert table["Weakest dimension"].iloc[0] == "Illumination (weak match)"
    assert "Illumination" not in table["Weakest dimension"].iloc[1]   # a dimension without data is not ranked


# ---------------------------------------------------------------- written atlas
def _synthetic_context():
    from src import analysis

    return analysis.build_context(synthetic.build_dataset(n_clusters=30, tiles_per_cluster=20, seed=1,
                                                               with_zoom=True))


def test_written_atlas_has_one_section_per_target():
    from src import analysis

    ctx = _synthetic_context()
    text = report.written_atlas(ctx, analysis.mission_dimension_weights("Rover Mobility"), "Rover Mobility")
    for name, body in zip(ctx.ds.targets["name"], ctx.ds.targets["body"], strict=True):
        assert f"### {name} ({body})" in text
    for part in ("# AnalogExplorer field atlas", "Mission profile: **Rover Mobility**", "## Summary",
                 "**Best distinct sites.**", "**Two sites that complete each other.**",
                 "**Can the ranking be trusted here?**", "**Below the surface.**",
                 "## What Earth can never reproduce", "## How the ranking works, in plain words"):
        assert part in text, part
    assert ".." not in text.replace("...", "") and "nan" not in text.lower().replace("nahuku", "")
    assert text.count(f"| Target | {report.BEST_SITE_COLUMN} |") == 1


def test_coords_text_names_the_hemispheres():
    assert report.coords_text(68.5, -92.5) == "68.5 N, 92.5 W"
    assert report.coords_text(-5.37, 137.81, 2) == "5.37 S, 137.81 E"


def test_atlas_command_writes_the_file(tmp_path, monkeypatch):
    from src import data_prep

    small = synthetic.build_dataset(n_clusters=30, tiles_per_cluster=20, seed=1)
    monkeypatch.setattr(data_prep, "load_default_dataset", lambda: small)
    out = tmp_path / "atlas.md"
    report.main(["--atlas", str(out), "--mission", "Solar Panels Deployment"])
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# AnalogExplorer field atlas") and "Solar Panels Deployment" in text
    assert "**Below the surface.**" not in text            # this dataset has no subsurface layer


def test_no_ice_prospecting_where_no_ice_is_documented():
    sims = {d: 0.99 for d in config.DIMENSIONS}
    log_sim = np.log(np.array([sims[d] for d in config.VAR_DIMENSIONS]))
    weights = {d: 1.0 for d in config.DIMENSIONS}
    with_ice = report.site_test_plan(log_sim, weights, "Mars", no_ice_reason=report.target_ice_note("mars_arcadia"))
    without = report.site_test_plan(log_sim, weights, "Mars", no_ice_reason=report.target_ice_note("mars_gale"))
    assert config.ICE_PROSPECTING_ACTIVITY in with_ice["Representative tests"]
    assert config.ICE_PROSPECTING_ACTIVITY not in without["Representative tests"]
    assert any(i.startswith("Ice prospecting: no ice") for i in without["Not representative here"])
    assert report.target_ice_note("unknown target") is None


def test_credibility_caveats_flag_a_reversed_cross_check():
    from types import SimpleNamespace

    targets = pd.DataFrame({"name": ["Moon site", "Mars site"], "body": ["Moon", "Mars"]})
    known = pd.DataFrame({"name": ["Small crater", "Big field"], "extent_km": [1.2, 300.0]})
    test = SimpleNamespace(n_sites=2, mean_rank=0.30)
    cred = SimpleNamespace(test=test, props=[("Topography",), ("Topography",)],
                           control=np.array([0.10, 0.20]), control_target=1)
    text = " ".join(report.credibility_caveats(cred, known, targets, p_holm=0.16))
    assert "would not pass" in text and "not show that the ranking is specific" in text
    assert "Small crater" in text and "Big field" not in text


# ------------------------------------------------------------ regressions of the 14 Nov 2026 audit
def test_labels_keep_their_units_in_sentences():
    assert report.lower_first("Ground below 0 C at depth") == "ground below 0 C at depth"
    assert report.property_label("ice_proxy") == "ground below 0 C at depth"
    assert report.property_label("Topography") == "Topography"


def test_equal_medians_are_not_reported_as_a_gap():
    levels = config.QUANTILE_LEVELS
    k = len(config.VARIABLES)
    target = np.ones((k, levels.size))
    site = target.copy()
    site[:, :3] = 0.2                                     # same median, lower tail differs
    profile = report.profile_table(site, target, _log_sim(np.full(k, 0.9)), levels)
    assert not profile["Key discrepancy"].str.contains("site 1.00 vs target 1.00").any()
    assert profile["Key discrepancy"].str.contains("same median as the target").all()
    exact = report.profile_table(target, target, np.zeros(k), levels)
    assert exact["Key discrepancy"].str.contains("identical to the target").all()


def test_candidate_table_shows_the_relative_range_as_given():
    log_score = np.log(np.array([0.2, 0.4, 0.1]))
    df = report.candidate_table(np.array([1, 0]), np.zeros(3), np.zeros(3), log_score, np.zeros(3), ["a"],
                                np.zeros(3, int), np.zeros(3), np.array([0.0, 0.5]),
                                interval=np.array([[0.8, 1.0], [0.3, 0.7]]))
    assert df["Relative score 90% range"].tolist() == ["0.80 - 1.00", "0.30 - 0.70"]


def test_uncovered_variables_lists_only_weighted_ones():
    """A variable the mission ignores, or that Earth cannot match, was announced as 'not covered'."""
    k = len(config.VARIABLES)
    uncovered = np.zeros(k, dtype=bool)
    uncovered[[2, 5, 6]] = True
    weights = np.full(k, 1.0)
    weights[[2, 5]] = 0.0                                 # out of range / not in the mission
    assert report.uncovered_variables(uncovered, weights) == [config.VARIABLES[6].label]


def test_credibility_table_explains_each_rank():
    from types import SimpleNamespace

    levels = config.QUANTILE_LEVELS
    k = len(config.VARIABLES)
    slope, frozen = config.VARIABLE_NAMES.index("slope_deg"), config.VARIABLE_NAMES.index("ice_proxy")
    target_q = np.ones((k, levels.size))
    target_q[slope] = 0.4
    tile_q = np.ones((2, k, levels.size))
    tile_q[0, slope] = 1.24
    known = pd.DataFrame({"name": ["Small crater", "Frozen island", "Islet", "Chemistry site"],
                          "extent_km": [1.2, 150.0, 15.0, 20.0]})
    cred = SimpleNamespace(
        props=[("Topography",), ("ice_proxy",), ("Topography",), ()], tiles=np.array([0, 1, -1, 0]),
        dist_km=np.array([68.0, 30.0, 209.0, 5.0]), ranks=np.array([0.72, 0.45, np.nan, np.nan]),
        control=np.array([0.10, 0.45, np.nan, np.nan]), worst_var=np.array([slope, frozen, -1, -1]),
        worst_sim=np.array([0.2, 1.0, np.nan, np.nan]), top_tie=np.array([0.0, 0.45, np.nan, np.nan]),
        at_top=np.array([False, True, False, False]))
    table = report.credibility_table(cred, known, tile_q, target_q, levels)
    assert table["Known analog"].tolist() == ["Small crater", "Frozen island", "Islet"]     # untested site left out
    why = table[report.WHY_COLUMN].tolist()
    assert "Largest gap: slope (s = 0.20), site 1.24 deg vs target 0.40 deg" in why[0] and "diluted" in why[0]
    assert "tied for first place" in why[1] and "45%" in why[1] and "ground below 0 C at depth" in why[1]
    assert why[2].startswith("Not ranked: no land tile within 150 km")
    assert "n/a" in report.markdown_table(table)          # a missing rank is not printed as 'nan'


# ------------------------------------------------------------ weak sites, Sun height, twin targets
def test_weak_best_site_is_said_plainly():
    assert report.weak_best_site_note(0.61) is None
    note = report.weak_best_site_note(0.22)
    assert "No single place on Earth" in note and "22 %" in note


def test_credibility_caveats_name_the_dominant_gap():
    from types import SimpleNamespace

    targets = pd.DataFrame({"name": ["Moon site", "Mars site"], "body": ["Moon", "Mars"]})
    known = pd.DataFrame({"name": ["a", "b", "c"], "extent_km": [30.0, 30.0, 30.0]})
    rough = config.VARIABLE_NAMES.index("roughness_m")
    cred = SimpleNamespace(test=SimpleNamespace(n_sites=3, mean_rank=0.4), props=[("Topography",)] * 3,
                           control=None, control_target=None, ranks=np.array([0.3, 0.5, np.nan]),
                           worst_var=np.array([rough, rough, -1]), at_top=np.array([False, False, False]))
    text = " ".join(report.credibility_caveats(cred, known, targets))
    assert "for 2 of the 2 ranked sites" in text and "roughness" in text


def test_twin_target_sentences_name_the_near_duplicates():
    overlap = np.array([[1.0, 0.75, 0.2], [0.75, 1.0, 0.3], [0.2, 0.3, 1.0]])
    names = ["Pole A", "Pole B", "Plain"]
    text = report.twin_target_sentences(names, overlap)
    assert len(text) == 1 and "Pole A and Pole B" in text[0] and "overlap by 75 %" in text[0]
    assert report.twin_target_sentences(names, overlap, only=2) == []
    one = report.twin_target_sentences(names, overlap, only=1)
    assert len(one) == 1 and "Pole B points to mostly the same places on Earth as: Pole A (75 %)" in one[0]
    assert report.p_text(0.0001) == "p < 0.001" and report.p_text(0.0412) == "p = 0.041"


def test_lighting_calendar_names_the_best_months_and_the_polar_night():
    sims = np.array([np.nan, 0.1, 0.8, 0.5, 0.2, 0.1, 0.1, 0.2, 0.6, 0.9, 0.3, np.nan])
    text = report.lighting_calendar_sentence(sims)
    assert "Mar, Sep, Oct" in text and "0.60 to 0.90" in text and "does not rise during 2 months" in text
    assert report.lighting_calendar_sentence(np.full(12, np.nan)) is None
    table = report.lighting_calendar_table(sims)
    assert table.shape == (1, 12) and table.columns.tolist()[:2] == ["Jan", "Feb"]
    assert np.isnan(table["Jan"].iloc[0]) and table["Oct"].iloc[0] == 0.9


# ------------------------------------------------------------ overview, data chain, place names (15 Nov 2026)
def test_data_chain_table_covers_every_variable():
    table = report.data_chain_table()
    assert len(table) == len(config.VARIABLES) and set(config.VARIABLE_SOURCES) == set(config.VARIABLE_NAMES)
    assert table["Dimension"].tolist() == list(config.VAR_DIMENSIONS)
    assert not (table[["Earth (every land tile)", "Moon targets", "Mars targets"]] == "").any().any()


def test_overview_table_matches_the_atlas_summary():
    """The app's overview and the atlas summary come from the same rows: same sites, same figures."""
    from src import analysis

    ctx = _synthetic_context()
    dim_w = analysis.mission_dimension_weights("Rover Mobility")
    creds, p_holm = analysis.credibility_all(ctx)
    table = report.overview_table(ctx, dim_w, creds, p_holm)
    assert len(table) == len(ctx.ds.targets) and report.BEST_SITE_COLUMN in table.columns
    atlas = report.written_atlas(ctx, dim_w, "Rover Mobility")
    for row in table.itertuples(index=False):
        assert f"| {row[0]} | {row[1]} | {row[2]} |" in atlas


def test_candidate_table_names_the_places():
    places = np.array(["Kunlun Mountains, China", "", "Sahara"], dtype=object)
    df = report.candidate_table(np.array([2, 1]), np.zeros(3), np.zeros(3), np.log(np.array([0.2, 0.4, 0.1])),
                                np.zeros(3), ["a"], np.zeros(3, int), np.zeros(3), np.array([0.0, 0.5]), places=places)
    assert df[report.PLACE_COLUMN].tolist() == ["Sahara", "-"]


def test_engine_diagram_follows_the_configuration(monkeypatch):
    """Page 1 diagram of the engine: the steps from the data to the ranking, thresholds read from config (a changed
    threshold changes the diagram), formulas with balanced LaTeX delimiters, no control character."""
    steps = report.engine_steps()
    titles = [title for title, _, _ in steps]
    assert titles[0] == "Open data" and titles.index("Similarity") < titles.index("Score") < len(titles)
    text = " ".join(detail for _, detail, _ in steps)
    assert f"{config.REACHABLE_MAX_PCT:.0%}" in text and f"z = {config.HUBER_DELTA:g}" in text
    assert f"{config.DEFAULT_RADIUS_KM:.0f} km" in text
    for title, detail, where in steps:
        assert detail.count("$") % 2 == 0 and where, title
        assert all(ch.isprintable() for ch in detail), title          # e.g. no '\r' from an unescaped '\right'
    monkeypatch.setattr(config, "NOVELTY_LAMBDA_KM", 123.0)
    assert "d / 123" in " ".join(detail for _, detail, _ in report.engine_steps())
    assert len(report.ENGINE_OUTPUTS) == 5
