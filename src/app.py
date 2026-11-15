r"""AnalogExplorer - Streamlit interface.

UI only: all calculations live in math_engine.py, data comes from data_prep.py.
Run from the repository root:  .\venv\Scripts\Activate.ps1 and streamlit run src/app.py

Caching
-------
- get_context()  : @st.cache_resource. The dataset is large and read-only, so it is shared
                   without copies (st.cache_data would copy it on every rerun).
- get_w1() etc.  : @st.cache_data on small derived arrays keyed by hashable parameters.
Widget values live in st.session_state["cfg"], not in widget keys, because Streamlit drops
the state of widgets that are not rendered on the current page.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # lets `streamlit run src/app.py` import the `src` package
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import analysis, charts, config, data_prep, report  # noqa: E402
from src import math_engine as me  # noqa: E402
from src.analysis import Context  # noqa: E402

PAGES = (   # (title, url path, icon): numbered, because the texts refer to "page 5"
    ("1. Home & Overview", "home", ":material/home:"),
    ("2. Target & Mission Selection", "target", ":material/my_location:"),
    ("3. Multi-Dimensional Profile", "profile", ":material/radar:"),
    ("4. Complementary Ensemble", "ensemble", ":material/hub:"),
    ("5. Limits Report & Site Sheet", "limits", ":material/fact_check:"),
    ("6. Earth Atlas (reverse search)", "atlas", ":material/public:"),
)
NAV_KEY = "nav"   # st.session_state[NAV_KEY]: url path -> page of this session (main), for the links between pages


def nav_page(path: str) -> st.Page:
    """Page of the current session's navigation. Per session, never module-wide: each page is bound to its
    session's choices, and a shared dict would let concurrent visitors run each other's pages."""
    return st.session_state[NAV_KEY][path]
MODES = ("Mission Mode", "Expert Mode")
ROBUSTNESS_BOX = "robustness_box"   # widget key of the robustness checkbox (page 3)
VAR_LABELS = [v.label for v in config.VARIABLES]


def _stretch(func) -> dict:
    """Full-width keyword accepted by this Streamlit version (the API changed over time)."""
    params = inspect.signature(func).parameters
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def show_table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, hide_index=True, **_stretch(st.dataframe), **kwargs)


def show_figure(fig) -> None:
    st.plotly_chart(fig, **_stretch(st.plotly_chart))


SIMILARITY_BAR = st.column_config.ProgressColumn("Similarity", min_value=0.0, max_value=1.0, format="%.2f")


# ================================================================= data & caches
def deployed_version() -> tuple:
    """(name, size, mtime) of the code and dataset files: changes whenever a new version is deployed."""
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    folder = data_prep.find_dataset_dir()
    if folder is not None:
        files += sorted(p for p in Path(folder).iterdir() if p.is_file())
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files)


@st.cache_resource
def _version_seen() -> dict:
    return {"version": None}


def clear_caches_on_new_version() -> None:
    """Streamlit Community Cloud reloads the code of a new commit without restarting the server, so objects
    cached by the previous version (the dataset context, results) would outlive it and break the new code
    (15 Nov 2026: 'Context' without 'place' after the place names were deployed). Start again from empty caches."""
    version, seen = deployed_version(), _version_seen()
    if seen["version"] is not None and seen["version"] != version:
        st.cache_data.clear()
        st.cache_resource.clear()
        seen = _version_seen()
    seen["version"] = version


@st.cache_resource(show_spinner="Loading dataset...")
def get_context() -> Context:
    return analysis.build_context(data_prep.load_default_dataset())


@st.cache_data(show_spinner=False)
def get_reachability(target_idx: int) -> me.Reachability:
    """Which variables of the target Earth can match at all (independent of weights)."""
    ctx = get_context()
    return analysis.target_reachability(ctx, get_w1(target_idx))


@st.cache_data(show_spinner=False)
def get_w1(target_idx: int) -> np.ndarray:
    """Per-variable Wasserstein distances of every tile to the target (independent of weights)."""
    ctx = get_context()
    return me.wasserstein1(ctx.ds.quantiles, ctx.ds.target_quantiles[target_idx])


@st.cache_data(show_spinner="Running Monte-Carlo...")
def get_robustness(target_idx: int, weights: tuple, tol_scale: float) -> np.ndarray:
    sigma = analysis.tolerances(get_context(), tol_scale)
    return me.robustness(get_w1(target_idx), sigma, np.asarray(weights), alpha=config.DIRICHLET_ALPHAS,
                         n_draws=config.ROBUSTNESS_DRAWS, top_frac=config.ROBUSTNESS_TOP_FRAC,
                         drop_prob=config.ROBUSTNESS_DROP_PROB, tile_weights=get_context().area)


@st.cache_data(show_spinner=False)
def get_score_interval(target_idx: int, weights: tuple, tol_scale: float, tiles: tuple) -> np.ndarray:
    """Range of the relative score (1 = best Earth tile of the same draw) of the shown sites, under the same
    weight perturbation as the robustness."""
    sigma = analysis.tolerances(get_context(), tol_scale)
    return me.score_interval(get_w1(target_idx), sigma, np.asarray(weights), np.asarray(tiles),
                             alpha=config.DIRICHLET_ALPHAS, n_draws=config.ROBUSTNESS_DRAWS,
                             q=config.SCORE_INTERVAL, drop_prob=config.ROBUSTNESS_DROP_PROB, relative=True)


@st.cache_data(show_spinner="Searching complementary sets...")
def get_ensemble(target_idx: int, weights: tuple, tol_scale: float, radius_km: float, enforce_min: bool):
    """Candidate pool and the best ensembles for k = 1..3."""
    return analysis.complementary_sets(get_context(), get_w1(target_idx), np.asarray(weights), tol_scale,
                                       radius_km, enforce_min)


@st.cache_data(show_spinner="Running the credibility check for every target...")
def get_credibility_holm(tol_scale: float) -> np.ndarray:
    """Credibility p-value of every target after a Holm correction for testing them all."""
    ctx = get_context()
    return analysis.credibility_all(ctx, tol_scale, {t: get_w1(t) for t in range(len(ctx.ds.targets))})[1]


@st.cache_data(show_spinner="Comparing every Earth tile with every target...")
def get_atlas(dim_weights: tuple, tol_scale: float) -> analysis.Atlas:
    """Every tile against every target (independent of the selected target)."""
    ctx = get_context()
    return analysis.atlas(ctx, dict(dim_weights), tol_scale, {t: get_w1(t) for t in range(len(ctx.ds.targets))})


@st.cache_data(show_spinner="Writing the atlas...")
def get_atlas_text(dim_weights: tuple, mission_label: str) -> bytes:
    """Written atlas of every target for these dimension weights (default advanced parameters)."""
    return report.written_atlas(get_context(), dict(dim_weights), mission_label).encode("utf-8")


# ================================================================== session state
def get_cfg() -> dict:
    """User choices, kept across pages."""
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = {
            "target_id": None, "mode": MODES[0], "use_case": next(iter(config.MISSION_PROFILES)),
            "expert": {d: 0.5 for d in config.DIMENSIONS}, "novelty_mix": 0.0, "tol_scale": 1.0,
            "radius_km": config.DEFAULT_RADIUS_KM, "k": 2, "site_rank": 1, "enforce_min": False,
            "robustness": True, "atlas_min": int(100 * config.STATUS_PARTIAL_CLOSER), "atlas_place": None,
            "selection": None,
        }
    return st.session_state["cfg"]


def current_dimension_weights(c: dict) -> dict[str, float]:
    """Raw dimension weights from the mission profile or the expert sliders."""
    dim_w = analysis.mission_dimension_weights(c["use_case"]) if c["mode"] == MODES[0] else dict(c["expert"])
    if sum(dim_w.values()) <= 0:  # all sliders at zero: fall back to equal weights
        dim_w = {d: 1.0 for d in config.DIMENSIONS}
    return dim_w


def current_weights(c: dict, available: np.ndarray | None = None) -> np.ndarray:
    """Variable weights (sum = 1); variables not available for the target get zero."""
    return me.variable_weights_from_dimensions(current_dimension_weights(c), config.VAR_DIMENSIONS, available)


def target_available(ctx: Context, t: int) -> np.ndarray:
    """(K,) bool: variables measured for target t."""
    return me.available_variables(ctx.ds.target_quantiles[t])


def scoring_mask(ctx: Context, t: int) -> np.ndarray:
    """(K,) bool: variables that enter the score: measured for the target and within Earth's range."""
    return analysis.scoring_mask(ctx.ds.target_quantiles[t], get_reachability(t))


def earth_pct(ctx: Context, c: dict, w: np.ndarray, log_scores: np.ndarray) -> np.ndarray:
    """Absolute percentile of sites against random Earth-Earth pairs (same weights and tolerances)."""
    return me.earth_percentile(log_scores, ctx.pair_gaps, analysis.tolerances(ctx, c["tol_scale"]), w)


def understudied_highlight(ctx: Context, c: dict, w: np.ndarray, rk: me.Ranking, sites: np.ndarray) -> None:
    """Best distinct site that is not a well-known analog (the challenge asks to look beyond them)."""
    tile = analysis.best_understudied(ctx, sites)
    if tile is None:
        st.info(f"No under-studied site among the distinct candidates (all within {config.UNDERSTUDIED_MIN_KM:.0f} km "
                "of a catalogued analog or in Antarctica, the Atacama, Iceland or Hawaii).")
        return
    rank = int(np.flatnonzero(sites == tile)[0]) + 1
    rel = np.exp(rk.log_score[tile] - rk.log_score.max())
    pct = 100 * earth_pct(ctx, c, w, rk.log_score[[tile]])[0]
    nearest = ctx.ds.known_analogs["name"].iloc[int(ctx.known_idx[tile])]
    url = report.worldview_url(ctx.ds.tile_lat[tile], ctx.ds.tile_lon[tile])
    st.success(f"**Best under-studied site: {site_label(ctx, tile)}**, rank #{rank} among distinct sites, relative "
               f"score {rel:.2f}, closer to the target than {pct:.0f} % of Earth pairs. It lies "
               f"{ctx.known_dist_km[tile]:.0f} km from the nearest catalogued analog ({nearest}) and outside the "
               f"famous analog regions. [See it in NASA Worldview]({url})")


def excluded_note(ctx: Context, t: int) -> str:
    """Sentence naming the variables left out of the score because Earth cannot match them."""
    reach = get_reachability(t)
    names = [VAR_LABELS[k] for k in np.flatnonzero(~reach.reachable)]
    return ("Out of Earth's range, excluded from the score and reported on page 5: " + ", ".join(names) + "."
            if names else "")


def mission_name(c: dict) -> str:
    return c["use_case"] if c["mode"] == MODES[0] else "Expert weights"


def reset_site_on_new_question(c: dict) -> None:
    """A new target or mission starts again from its best site (not the first visit of page 2)."""
    selection = (c["target_id"], mission_name(c))
    if c.get("selection") not in (None, selection):
        c["site_rank"] = 1
    c["selection"] = selection


def active_target(c: dict, ctx: Context) -> int:
    ids = ctx.ds.targets["target_id"].tolist()
    if c["target_id"] not in ids:
        c["target_id"] = ids[0]
    return ctx.ds.target_index(c["target_id"])


def target_info(ctx: Context, t: int) -> tuple[str, str]:
    """(name, body) of target t."""
    row = ctx.ds.targets.iloc[t]
    return str(row["name"]), str(row["body"])


def compute(c: dict):
    """Everything that depends on the current choices; cheap enough to run on every rerun."""
    ctx = get_context()
    t = active_target(c, ctx)
    a = analysis.analyse_target(ctx, t, current_dimension_weights(c), c["tol_scale"], c["novelty_mix"],
                                c["radius_km"], w1=get_w1(t), reach=get_reachability(t))
    return ctx, t, a.weights, a.ranking


def weights_key(w: np.ndarray) -> tuple:
    return tuple(round(float(x), 6) for x in w)


WORLDVIEW_LINK = st.column_config.LinkColumn("NASA Worldview", display_text="Open imagery")
STATUSES = ("Strong match", "Partial match", "Weak match", report.NOT_EVALUATED)
STATUS_CHIP = st.column_config.MultiselectColumn("Status", options=STATUSES, color=["green", "orange", "red", "gray"])
PROFILE_COLUMNS = {"Similarity": SIMILARITY_BAR, "Status": STATUS_CHIP,
                   report.CLOSER_COLUMN: st.column_config.NumberColumn(format="%.0f %%")}
CANDIDATE_COLUMNS = {    # display format only: the values (and the CSV export) are unchanged
    "NASA Worldview": WORLDVIEW_LINK,
    "Lat": st.column_config.NumberColumn(format="%.1f"), "Lon": st.column_config.NumberColumn(format="%.1f"),
    "Relative score": st.column_config.NumberColumn(format="%.2f"),
    report.CLOSER_COLUMN: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f %%"),
    "Top %": st.column_config.NumberColumn(format="%.2f %%"),
    "Novelty": st.column_config.NumberColumn(format="%.2f"),
    "Distance (km)": st.column_config.NumberColumn(format="%.0f"),
    "Robustness": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
}


def status_chips(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df whose 'Status' is shown as a coloured chip (the column type needs a list per cell)."""
    out = df.copy()
    out["Status"] = [[s] for s in out["Status"]]
    return out


def data_banner(ctx: Context) -> None:
    """Warning on every page when the dataset is a sample; the full list of sources is on page 1."""
    limit = ctx.ds.meta.get("sample_limit")
    if limit:
        st.warning(f"Sample of {limit} tiles only, not the whole Earth.")


def data_sources(ctx: Context) -> None:
    """Data sources and place names (page 1, folded): said once rather than at the top of every page."""
    meta = ctx.ds.meta
    with st.expander("Data sources", icon=":material/database:"):
        if meta.get("source") == "real":
            st.markdown("Real data: " + "; ".join(f"{k}: {v}" for k, v in meta.get("sources", {}).items()) + ".")
        if meta.get("places"):
            st.caption(f"Place names: {meta['places']['source']}; country and physical region of the tile centre, "
                       "a reading aid outside the score (country lines are Natural Earth's de facto boundaries; no "
                       "country is shown in areas it lists as disputed).")
    data_banner(ctx)


def engine_diagram(card_px: int = 215) -> None:
    """Page 1, folded: the steps of the engine as numbered cards with their formulas, then the results (no
    computation). Cards have a fixed width and wrap: as many per row as the screen holds, one on a phone."""
    steps = report.engine_steps()
    with st.expander("How the engine works", icon=":material/account_tree:"):
        with st.container(horizontal=True, wrap=True):
            for i, (title, detail, where) in enumerate(steps, start=1):
                with st.container(border=True, width=card_px):
                    st.markdown(f"**{i}. {title}** " + ("→" if i < len(steps) else "↓"))
                    st.markdown(detail)
                    st.caption(where)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**Results:**", width="content")
            for text, where in report.ENGINE_OUTPUTS:
                st.badge(f"{text} ({where})", color="blue")
        st.caption("The same code computes every page; thresholds come from the project's configuration.")


def context_line(ctx: Context, c: dict, t: int | None = None) -> None:
    """Current target (when the page depends on it) and mission, with a link back to page 2."""
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        if t is not None:
            name, body = target_info(ctx, t)
            st.badge(f"Target: {name} ({body})", icon=":material/my_location:")
        st.badge(f"Mission: {mission_name(c)}", icon=":material/rocket_launch:", color="violet")
        st.page_link(nav_page("target"), label="Change (page 2)", icon=":material/tune:")


GLOSSARY = (
    ("Tile", "A 1 x 1 degree cell of land (about 110 km), the unit of the global screening."),
    ("Target", "One of the Moon or Mars sites the Earth is compared with."),
    ("Dimension", "A group of variables: " + ", ".join(config.DIMENSIONS) + "."),
    ("Similarity", "0 to 1 for each variable: 1 = the site has the target's distribution of that variable "
                   "(Wasserstein distance, in tolerances)."),
    ("Tolerance", "Typical gap between two random places on Earth's land, for each variable: the unit of the score."),
    ("Relative score", "1 = the best Earth tile for this target and mission."),
    ("Closer than X % of Earth pairs", "50 % = the site is as close to the target as two typical places on Earth "
                                       "are to each other."),
    ("Strong / partial match", f"Closer than {config.STATUS_STRONG_CLOSER:.0%} / "
                               f"{config.STATUS_PARTIAL_CLOSER:.0%} of Earth pairs."),
    ("Top %", "Share of Earth's land area that scores at least as well."),
    ("Robustness", f"Probability of staying in the best {config.ROBUSTNESS_TOP_FRAC:.0%} of Earth's land when the "
                   "mission weights are perturbed."),
    ("Novelty", f"0 at a catalogued analog, towards 1 far from all of them (1 - exp(-d / "
                f"{config.NOVELTY_LAMBDA_KM:.0f} km))."),
    ("Under-studied", f"More than {config.UNDERSTUDIED_MIN_KM:.0f} km from any catalogued analog and outside "
                      "Antarctica, the Atacama, Iceland and Hawaii."),
    ("Coverage", "Weighted share of the target's conditions that at least one site of a set reproduces."),
)


def sidebar(ctx: Context) -> None:
    """Below the navigation: glossary and data line."""
    with st.sidebar:
        with st.popover("Glossary", icon=":material/menu_book:", width="stretch"):
            st.markdown("\n".join(f"* **{term}**: {text}" for term, text in GLOSSARY))
        st.caption(f"Data: {ctx.ds.meta.get('source', 'unknown')}, {ctx.ds.n_tiles:,} Earth tiles. "
                   "AnalogExplorer, NASA Space Apps Challenge 2026.")


def site_label(ctx: Context, tile: int, rank: int | None = None) -> str:
    """'#1 Kunlun Mountains, China (36.5 N, 86.5 E)': place name when the dataset has one, and coordinates."""
    prefix = f"#{rank} " if rank else ""
    return prefix + report.site_name(ctx.place[tile], ctx.ds.tile_lat[tile], ctx.ds.tile_lon[tile])


def zoom_view(ctx: Context, c: dict, t: int, w: np.ndarray, tile_log_sim: np.ndarray, tile: int):
    """(TileZoom, ZoomResult) of a tile for the current target and weights; (None, None) without stored zoom."""
    tz = analysis.tile_zoom(ctx, tile)
    if tz is None:
        return None, None
    return tz, analysis.zoom_similarity(ctx, t, tz.quantiles, w, tile_log_sim, c["tol_scale"])


def zoom_section(ctx: Context, c: dict, t: int, w: np.ndarray, tile_log_sim: np.ndarray, tile: int) -> None:
    """Fine-scale zoom of the inspected site: where inside the tile is the best match?"""
    st.subheader("Zoom: where inside this site?")
    tz, res = zoom_view(ctx, c, t, w, tile_log_sim, tile)
    if res is None or res.order.size == 0:
        st.info("No fine-scale zoom is stored for this site. Zooms are precomputed offline for the sites shown "
                "with the default settings of each mission profile (python -m src.data_prep --layers).")
        return
    ds = ctx.ds
    bounds = tz.bounds()
    best = int(res.order[0])
    table = report.zoom_table(bounds, tz.quantiles, res.similarity, res.order, ds.levels)
    target_q = ds.target_quantiles[t][list(analysis.TOPO_IDX)]
    cols = st.columns(3)
    cols[0].metric("Best sub-area (centre)", f"{table['Lat'].iloc[0]:.2f}, {table['Lon'].iloc[0]:.2f}",
                   help=f"About {table['Size (km)'].iloc[0]} km.")
    cols[1].metric("Topographic similarity there", f"{res.similarity[best]:.2f}",
                   delta=f"{res.similarity[best] - res.tile_similarity:+.2f} vs the whole site", delta_color="off",
                   help=f"Whole site: {res.tile_similarity:.2f}. Same target profile, tolerances and weights. A "
                        "negative gap means no single sub-area reproduces the target's terrain as well as the "
                        "site taken as a whole.")
    cols[2].metric("Sub-areas compared", f"{res.order.size} of {res.similarity.size}",
                   help="A sub-area needs at least half of its pixels on land.")
    if charts.HAS_PLOTLY:
        frac = bounds - np.array([tz.lat0, tz.lat0, tz.lon0, tz.lon0])
        target_id = ds.targets["target_id"].iloc[t]
        show_figure(charts.zoom_figure(
            tz.dem, tz.pixel_size_m(), ds.target_dems.get(target_id), frac, res.similarity, tz.grid, best,
            target_px_m=data_prep.TOPO_PIXEL_M,
            titles=("Earth site (shaded relief)", "Similarity of each sub-area", f"{target_info(ctx, t)[0]}")))
    show_table(table, column_config={"NASA Worldview": WORLDVIEW_LINK,
                                     "Topographic similarity": SIMILARITY_BAR})
    st.caption(f"Target terrain: median slope {report.median_of(target_q[0], ds.levels):.1f} deg, median roughness "
               f"{report.median_of(target_q[1], ds.levels):.0f} m. The site is cut into sub-areas of about 11 km; "
               "slope and roughness use the same elevation data, the same ~463 m pixel and the same operators as "
               "the global screening, and are compared with the same target profile. Only the terrain is zoomed: "
               "temperature, water and sunlight stay at the scale of the whole site (climate grids of 0.5 to 1 "
               "degree). A sub-area holds a few hundred pixels, so its figures are noisier than the site's. "
               "The three panels share the same ground scale.")


def subsurface_section(ctx: Context, t: int, rk: me.Ranking) -> None:
    """Subsurface case studies: lava tubes on Earth against the target's pit or cave."""
    st.subheader("Below the surface (case studies)")
    feature = config.SUBSURFACE_TARGETS.get(ctx.ds.targets["target_id"].iloc[t])
    sites = ctx.ds.subsurface
    if feature is None or feature.temperature_k is None or sites is None:
        st.write("No pit or cave is documented for this target in this version, so its subsurface is not "
                 "evaluated. Catalogues exist (LROC Lunar Pit Atlas, USGS Mars cave-candidate catalogue) but have "
                 "not been searched around this target.")
        return
    st.markdown(f"**{feature.name}.** {feature.geometry} {feature.temperature_note}")
    tiles, _ = me.nearest_tiles(ctx.tree, sites["lat"].to_numpy(), sites["lon"].to_numpy(), config.CREDIBILITY_MAX_KM)
    matched = tiles >= 0
    top, under = np.full(len(sites), np.nan), np.zeros(len(sites), dtype=bool)
    top[matched] = analysis.top_share(ctx, rk.log_score, tiles[matched])
    under[matched] = ctx.understudied[tiles[matched]]
    show_table(report.subsurface_table(sites, feature.temperature_k, top, under),
               column_config={"NASA Worldview": WORLDVIEW_LINK})
    st.caption(f"{config.CAVE_TEMPERATURE_RULE} The mean annual air temperature (NASA POWER, a 0.5 x 0.625 degree "
               "cell average that can differ by several K from the cave's own altitude) therefore estimates the "
               "temperature inside each lava tube; a small gap means the tube offers the thermal environment "
               "modelled inside the target. Case studies, outside the score: an Earth cave holds air, moisture and "
               "life, while the target is in vacuum, where heat moves by radiation and conduction only. 'Surface "
               "match' is the rank of the tile around the tube for the current target and mission (empty: no land "
               "tile within 150 km, as on small islands). "
               f"Sources: {feature.source} {config.LAVA_TUBE_SOURCE}.")


def known_layer(ctx: Context) -> tuple[np.ndarray, np.ndarray, list[str]]:
    k = ctx.ds.known_analogs
    return k["lat"].to_numpy(), k["lon"].to_numpy(), k["name"].tolist()


def ensemble_rows(ctx: Context, tiles: np.ndarray, res: me.EnsembleResult) -> pd.DataFrame:
    """One row per site of the set, with the variables it is the best of the set for."""
    best_for = {i: [] for i in range(len(tiles))}
    for var_idx, site_pos in enumerate(res.best_site_per_variable):
        if np.isfinite(res.best_per_variable[var_idx]):  # skip variables not evaluated for the target
            best_for[int(site_pos)].append(VAR_LABELS[var_idx])
    return pd.DataFrame({
        "Site": [chr(65 + i) for i in range(len(tiles))],
        report.PLACE_COLUMN: [str(ctx.place[i]) or "-" for i in tiles],
        "Lat": np.round(ctx.ds.tile_lat[tiles], 2), "Lon": np.round(ctx.ds.tile_lon[tiles], 2),
        "Nearest known analog": [ctx.ds.known_analogs["name"].iloc[int(ctx.known_idx[i])] for i in tiles],
        "Best of the set for": [", ".join(best_for[i]) or "-" for i in range(len(tiles))]})


# ======================================================================== pages
def page_home(ctx: Context) -> None:
    st.title("AnalogExplorer")
    st.markdown("#### Which places on Earth stand in best for permanent Moon base sites and Mars sites, dimension "
                "by dimension, and what can Earth not reproduce?")
    st.write("AnalogExplorer identifies, characterizes and links terrestrial analogs to planetary targets "
             "(permanent Moon base locations and Mars sites).")
    steps = (
        ("target", "1. Choose", "a Moon or Mars target and a mission profile.", "Start on page 2"),
        ("profile", "2. Explore", "the best distinct Earth sites, dimension by dimension (page 3), and the set of "
                                  "sites that together cover the target (page 4).", "Page 3"),
        ("limits", "3. Check", "what Earth cannot reproduce and the site sheet to download (page 5), or start from "
                               "any place on Earth (page 6).", "Page 5"),
    )
    for col, (path, step, text, link) in zip(st.columns(3), steps, strict=True):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{step}** {text}")
            st.page_link(nav_page(path), label=link, icon=":material/arrow_forward:")
    data_sources(ctx)
    engine_diagram()

    cols = st.columns(4, border=True)
    cols[0].metric("Earth tiles", f"{ctx.ds.n_tiles:,}")
    cols[1].metric("Targets", len(ctx.ds.targets))
    cols[2].metric("Dimensions", len(config.DIMENSIONS))
    cols[3].metric("Variables", len(ctx.ds.variables))

    st.subheader("How we answer the challenge")
    counts = report.challenge_counts()
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown("The official challenge text, item by item (4 objectives, 12 considerations):", width="content")
        for (status, n), color in zip(counts.items(), ("green", "orange", "red"), strict=True):
            st.badge(f"{n} {status.lower()}", color=color)
    # The summary stays in view (badges above); the 16 rows are folded so the page stays short.
    with st.expander(f"See the {sum(counts.values())} items, their status and the page that answers each",
                     icon=":material/checklist:"):
        st.markdown(report.markdown_table(report.challenge_table()))    # plain table: the long cells wrap
        st.caption("'Page' is the number of the view in the sidebar. The status says what this version does, "
                   "not how well: the limits are detailed on page 5.")

    st.subheader("Core methodology")
    st.markdown(
        "* **Characterize before ranking:** a single score hides discrepancies, so the main output is a "
        "profile by dimension (topography, thermal, water/ice, illumination, surface cover).\n"
        "* **Compare distributions, not averages:** each tile is described by quantiles; similarity uses the "
        "Wasserstein distance, scored with a non-compensatory geometric mean. The penalty grows with the square "
        f"of the gap up to {config.HUBER_DELTA} tolerances, then linearly (Huber), so one very poor variable does "
        "not decide the ranking alone.\n"
        "* **Earth counted by land area:** every share of Earth ('closer than X % of Earth pairs', 'top X %') is "
        "a share of land surface, so the many narrow tiles near the poles do not weigh more than their area.\n"
        "* **Complementary ensemble:** 2-3 sites that together cover the dimensions of a target.\n"
        "* **Scientific honesty:** structural limits, uncovered dimensions, robustness of the result and a "
        "credibility check against known analogs, with criteria fixed in advance.")


def page_target(ctx: Context, c: dict) -> None:
    st.title("Target & Mission Configuration")
    st.write("Select a planetary target, then a mission profile or your own dimension weights.")
    data_banner(ctx)
    targets = ctx.ds.targets
    left, right = st.columns(2)

    with left:
        st.subheader("1. Planetary target")
        bodies = list(dict.fromkeys(targets["body"]))
        active_target(c, ctx)  # makes sure c["target_id"] is valid
        current_body = targets.loc[targets["target_id"] == c["target_id"], "body"].iloc[0]
        body = st.selectbox("Celestial body", bodies, index=bodies.index(current_body))
        options = targets[targets["body"] == body]
        ids = options["target_id"].tolist()
        index = ids.index(c["target_id"]) if c["target_id"] in ids else 0
        names = dict(zip(options["target_id"], options["name"], strict=True))
        c["target_id"] = st.selectbox("Target site", ids, index=index, format_func=names.get)
        st.success(f"Selected target: {names[c['target_id']]}")
        st.caption(str(options.loc[options["target_id"] == c["target_id"], "note"].iloc[0]))

    with right:
        st.subheader("2. Mission profile")
        c["mode"] = st.selectbox("Mode", MODES, index=MODES.index(c["mode"]))
        if c["mode"] == MODES[0]:
            cases = list(config.MISSION_PROFILES)
            c["use_case"] = st.selectbox("Case study", cases, index=cases.index(c["use_case"]))
            levels = config.MISSION_PROFILES[c["use_case"]]
            st.write("Importance by dimension: " + ", ".join(f"**{d}** {lvl}" for d, lvl in levels.items()))
            st.caption(f"Why: {config.MISSION_RATIONALE.get(c['use_case'], '')} "
                       "(hypothesis to back with references)")
        else:
            for dim in config.DIMENSIONS:
                c["expert"][dim] = st.slider(f"{dim} weight", 0.0, 1.0, float(c["expert"].get(dim, 0.5)), 0.01)
            if sum(c["expert"].values()) <= 0:
                st.warning("All weights are zero: equal weights are used instead.")
    reset_site_on_new_question(c)

    t = active_target(c, ctx)
    available = target_available(ctx, t)
    w = current_weights(c, scoring_mask(ctx, t))
    status = np.where(~available, "not evaluated",
                      np.where(get_reachability(t).reachable, "measured", "out of Earth's range"))
    st.subheader("Resulting variable weights")
    show_table(pd.DataFrame({"Dimension": list(config.VAR_DIMENSIONS), "Variable": VAR_LABELS,
                             "Definition": [v.description for v in config.VARIABLES], "Weight": w,
                             "Target data": status}),
               column_config={"Weight": st.column_config.ProgressColumn("Weight", min_value=0.0,
                                                                        max_value=1.0, format="%.2f")})
    if (status != "measured").any():
        st.caption("Variables without data, or that no place on Earth can match, are left out of the score: "
                   "their weight goes to the other variables of the same dimension. Out-of-range variables are "
                   "reported with figures on page 5.")
    target_sources = ctx.ds.meta.get("target_sources", {}).get(c["target_id"])
    if target_sources:
        with st.expander("Where the target values come from"):
            show_table(pd.DataFrame({"Variable": [config.VARIABLE_BY_NAME[k].label for k in target_sources],
                                     "Source": list(target_sources.values())}))

    atlas = get_atlas(tuple(sorted(current_dimension_weights(c).items())), float(c["tol_scale"]))
    for sentence in report.twin_target_sentences(targets["name"].tolist(), atlas.top_overlap, only=t):
        st.info(sentence)

    with st.expander("Advanced parameters"):
        c["tol_scale"] = st.slider("Tolerance scale (x default sigma)", 0.5, 2.0, float(c["tol_scale"]), 0.1)
        st.caption("Default sigma: typical gap between two random places on Earth's land, per variable. A larger "
                   "scale is more permissive.")
        c["radius_km"] = st.slider("Minimum distance between distinct sites (km)", 50, 1000,
                                   int(c["radius_km"]), 50)


def verdict_explanation(test: me.CredibilityTest, crit: dict, body: str) -> str:
    """Plain-language reason of the credibility verdict, for the tooltip of its badge."""
    if not test.conclusive:
        return (f"Only {test.n_sites} known {body} analogs are matched to a tile; the rule fixed in advance needs at "
                f"least {crit['min_sites']} to conclude.")
    chance = "less than 0.1 %" if test.p_value < 0.001 else f"{100 * test.p_value:.0f} %"
    ranks = (f"On the properties they are famous for, the {test.n_sites} known {body} analogs rank on average in the "
             f"top {100 * test.mean_rank:.0f} % of Earth's land; random places on land rank in the top "
             f"{100 * test.null_mean:.0f} %. ")
    if test.passed:
        return (ranks + f"Random sets of places do as well in only {chance} of draws ({report.p_text(test.p_value)}), "
                f"below the {100 * crit['alpha']:.0f} % fixed in advance: the method finds the known analogs.")
    return (ranks + f"Random sets of places do as well in {chance} of draws ({report.p_text(test.p_value)}), above "
            f"the {100 * crit['alpha']:.0f} % fixed in advance: not clearly better than chance. Known analogs are "
            "often smaller than a tile (about 110 km) and were also chosen for access and logistics, so this does "
            "not disprove them.")


def credibility_section(ctx: Context, t: int, tol_scale: float) -> None:
    """Credibility check: known analogs ranked on the exact properties they are famous for."""
    crit = config.CREDIBILITY_TEST
    _, body = target_info(ctx, t)
    known = ctx.ds.known_analogs
    cred = analysis.credibility(ctx, t, tol_scale, {i: get_w1(i) for i in range(len(ctx.ds.targets))})
    test = cred.test

    st.caption(f"Protocol fixed in advance: each known {body} analog of the sourced catalogue is ranked among all "
               f"tiles on the exact properties its sources document (e.g. the Atacama on aridity, permafrost sites "
               f"on ground below 0 C at depth, Artemis training craters on topography). The mean percentile rank must "
               f"beat random places on land (p < {crit['alpha']} over {crit['n_draws']:,} random sets). Fewer than "
               f"{crit['min_sites']} sites: inconclusive. Revision (13 Nov 2026, after a first run): properties were "
               "first coded as whole dimensions; sites and sources are unchanged. Revision (14 Nov 2026): Ladakh and "
               "Lake Salda, documented Mars analogs found near sites first called under-studied, were added for the "
               "novelty axis only (not tested here). Revision (14 Nov 2026, after the results): ranks are shares of "
               "Earth's land area instead of counts of tiles, and places tied on a property share their average "
               "rank instead of all counting as ahead of each other; the results before and after are kept by the "
               "team. The goal is not to rediscover known sites but to check that the method is sound.")
    verdict = "PASS" if test.passed else "FAIL" if test.conclusive else "INCONCLUSIVE"
    cols = st.columns(4)
    with cols[0]:
        st.caption("Verdict", help=verdict_explanation(test, crit, body))    # "?" like the metrics beside it
        st.badge(verdict, color={"PASS": "green", "FAIL": "red"}.get(verdict, "gray"),
                 icon={"PASS": ":material/check_circle:", "FAIL": ":material/cancel:"}.get(verdict, ":material/help:"))
    cols[1].metric("Mean rank of known sites", f"top {100 * test.mean_rank:.0f} %",
                   help=f"Share of Earth's land ranking at least as well. Random places on land: top "
                        f"{100 * test.null_mean:.0f} %. {report.p_text(test.p_value)}, z = {test.z:.2f}.")
    cols[2].metric(f"In top {crit['top_frac']:.0%} (former criterion)", f"{test.n_in_top} / {test.n_sites}",
                   help=f"Exact binomial p = {test.binomial_p:.3f}.")
    cols[3].metric("Sanity test: target vs itself", "s = 1" if cred.self_ok else "FAILED")
    st.write(f"Known sites rank on average in the top **{100 * test.mean_rank:.0f} %** of Earth's land, random "
             f"places on land in the top {100 * test.null_mean:.0f} %: **{report.p_text(test.p_value)}**, effect "
             f"size z = {test.z:.2f}.")
    p_holm = float(get_credibility_holm(tol_scale)[t])
    st.markdown("\n".join(f"* {s}" for s in report.credibility_caveats(cred, known, ctx.ds.targets, p_holm)))
    st.warning(config.CREDIBILITY_CAVEAT)
    if not test.passed:
        st.info("Reported as is (protocol fixed in advance). Known analogs are often smaller than a tile and were "
                "also chosen for access and logistics, so a modest rank does not disprove them.")
    table = report.credibility_table(cred, known, ctx.ds.quantiles, ctx.ds.target_quantiles[t], ctx.ds.levels)
    st.markdown(report.markdown_table(table))     # plain table: the reasons are sentences and must wrap
    st.caption(f"{len(table)} of {len(known)} catalogued analogs are tested for {report.body_phrase(body)}; "
               "the others are famous for properties we do not model (e.g. chemistry) and only feed the novelty "
               f"axis. '{report.WHY_COLUMN}' is read from the data: the tested variable with the largest gap, ties "
               "(places that match the target exactly share the first place and take the middle rank of their tie), "
               "or a site smaller than a tile. "
               "Sources: Preston & Dartnell 2014, Foucher et al. 2021, NASA Artemis and ESA PANGAEA training "
               "sites (full list in the dataset's known_analogs.csv).")


def page_profile(ctx: Context, c: dict) -> None:
    st.title("Multi-Dimensional Profile")
    st.write("How well do the best distinct Earth sites match the target, dimension by dimension?")
    data_banner(ctx)
    context_line(ctx, c, active_target(c, ctx))
    c["novelty_mix"] = st.slider("Similarity (0) to novelty (1)", 0.0, 1.0, float(c["novelty_mix"]), 0.05,
                                 help="Shifts the ranking towards sites far from catalogued analogs.")
    ctx, t, w, rk = compute(c)
    ds = ctx.ds
    sites = rk.site_idx
    if sites.size == 0:
        st.error("No candidate site. Lower the minimum distance between sites in the advanced parameters (page 2).")
        return

    shown = sites[:config.TOP_SITES_SHOWN]
    if ROBUSTNESS_BOX in st.session_state:       # the box sits below the table: read its new state before using it
        c["robustness"] = bool(st.session_state[ROBUSTNESS_BOX])
    robustness = get_robustness(t, weights_key(w), c["tol_scale"]) if c["robustness"] else None
    interval = get_score_interval(t, weights_key(w), c["tol_scale"], tuple(int(i) for i in shown))
    table = report.candidate_table(
        shown, ds.tile_lat, ds.tile_lon, rk.log_score, ctx.novelty, ds.known_analogs["name"].tolist(),
        ctx.known_idx, ctx.known_dist_km, analysis.top_share(ctx, rk.log_score, shown), robustness, interval,
        earth_pct(ctx, c, w, rk.log_score[shown]), ctx.understudied, ctx.place)
    st.subheader("Best distinct sites")
    show_table(table, column_config=CANDIDATE_COLUMNS)
    st.caption("Relative score: 1 = best Earth tile for this target and mission. 'Closer than % of Earth pairs': "
               "50 % means the site is as close to the target as two typical places on Earth are to each other. "
               "'Top %': share of Earth's land area that scores at least as well. 'Robustness': probability of "
               "staying in the best 1 % of Earth's land when the mission weights are perturbed. "
               + excluded_note(ctx, t))
    weak = report.weak_best_site_note(float(earth_pct(ctx, c, w, rk.log_score[sites[:1]])[0]))
    if weak and c["novelty_mix"] == 0:
        st.warning(weak)
    understudied_highlight(ctx, c, w, rk, sites)
    c["robustness"] = st.checkbox("Compute robustness (Monte-Carlo on the weights)", value=c["robustness"],
                                  key=ROBUSTNESS_BOX,
                                  help="Probability of staying in the best 1 % of Earth's land when the mission "
                                       "weights are "
                                       "perturbed at three levels of uncertainty and, in a third of the draws, one "
                                       "variable is left out. The score range uses the same perturbation.")

    c["site_rank"] = int(np.clip(c["site_rank"], 1, len(shown)))
    pick = st.selectbox("Site to inspect", list(range(len(shown))), index=c["site_rank"] - 1,
                        format_func=lambda i: site_label(ctx, int(shown[i]), i + 1))
    c["site_rank"] = pick + 1
    tile = int(shown[pick])
    closer = me.closer_than_share(get_w1(t)[tile], ctx.pair_gaps)
    profile = report.profile_table(ds.quantiles[tile], ds.target_quantiles[t], rk.log_sim[tile], ds.levels, closer)

    # The same sections as before, in tabs instead of one long page: everything is still computed and shown.
    tab_profile, tab_dist, tab_zoom, tab_map, tab_cred = st.tabs(
        ["Profile by dimension", "Distributions", "Zoom inside the site", "Map & novelty", "Credibility check"])
    with tab_profile:
        left, right = st.columns(2)
        with left:
            st.subheader("Similarity by variable")
            if charts.HAS_PLOTLY:
                strong = me.similarity_at_closer_share(ctx.pair_gaps, analysis.tolerances(ctx, c["tol_scale"]),
                                                       config.STATUS_STRONG_CLOSER)
                show_figure(charts.radar(profile["Variable"].tolist(), profile["Similarity"].to_numpy(),
                                         strong=strong))
                st.caption(f"Dotted line: similarity above which a variable is a strong match (closer than "
                           f"{config.STATUS_STRONG_CLOSER:.0%} of Earth pairs); it differs from one variable to the "
                           "next.")
            else:
                st.bar_chart(profile.set_index("Variable")["Similarity"])
        with right:
            st.subheader("Similarity by dimension")
            show_table(status_chips(report.dimension_summary(rk.log_sim[tile], closer)),
                       column_config=PROFILE_COLUMNS)
            st.info("A weak dimension stays visible: the score is a geometric mean, so a good result elsewhere "
                    "cannot hide it.")
        show_table(status_chips(profile[["Dimension", "Variable", "Similarity", report.CLOSER_COLUMN, "Status",
                                         "Site (median)", "Target (median)"]]), column_config=PROFILE_COLUMNS)
        st.caption(f"Status: strong when the site is closer to the target than {config.STATUS_STRONG_CLOSER:.0%} "
                   f"of random pairs of Earth places are to each other, partial above "
                   f"{config.STATUS_PARTIAL_CLOSER:.0%} (50 % = one tolerance away from the target, the unit of the "
                   "score, for every variable).")

    with tab_dist:
        st.subheader("Where the distributions differ")
        if charts.HAS_PLOTLY:
            st.caption("Each panel compares the distribution of a variable inside the site with the target "
                       "(quantile curves). The shaded area is the Wasserstein distance used by the engine.")
            show_figure(charts.distribution_compare(ds.quantiles[tile], ds.target_quantiles[t], ds.levels,
                                                    VAR_LABELS, [v.unit for v in config.VARIABLES]))
        else:
            st.info("Plotly is not installed: no distribution figure.")

    with tab_zoom:
        zoom_section(ctx, c, t, w, rk.log_sim[tile], tile)

    with tab_map:
        score = np.exp(rk.log_score[sites])
        nov = ctx.novelty[sites]
        if charts.HAS_PLOTLY:
            st.subheader("Map of candidate sites")
            show_figure(charts.sites_map(ds.tile_lat[sites], ds.tile_lon[sites], score, highlight=[pick],
                                         known=known_layer(ctx), places=ctx.place[sites]))
            st.subheader("Similarity versus novelty")
            show_figure(charts.similarity_novelty_scatter(score, nov, me.pareto_front(score, nov), selected=pick))
        else:
            st.map(pd.DataFrame({"lat": ds.tile_lat[sites], "lon": ds.tile_lon[sites]}))

    with tab_cred:
        st.subheader("Credibility check: where do known analogs rank?")
        credibility_section(ctx, t, c["tol_scale"])


def page_ensemble(ctx: Context, c: dict) -> None:
    st.title("Complementary Ensemble")
    st.write("No single place replicates another world. Here is the set of Earth sites that, together, "
             "covers the target's dimensions best.")
    data_banner(ctx)
    c["k"] = st.slider("Ensemble size (k sites)", 1, 3, int(c["k"]))
    c["enforce_min"] = st.checkbox("Require minimum coverage on every weighted variable",
                                   value=c["enforce_min"],
                                   help=f"Only sets reaching {config.COVERED_THRESHOLD} on each weighted "
                                        "variable are eligible, when such a set exists.")
    ctx, t, w, rk = compute(c)
    context_line(ctx, c, t)
    ds = ctx.ds
    pool, sims, curve = get_ensemble(t, weights_key(w), c["tol_scale"], c["radius_km"], c["enforce_min"])
    res = curve[c["k"] - 1]
    tiles = pool[list(res.indices)]

    cols = st.columns(4)
    cols[0].metric(f"Coverage ({len(tiles)} sites)", f"{res.coverage:.2f}")
    cols[1].metric("Best single site", f"{res.best_single:.2f}")
    cols[2].metric("Gain from complementarity", f"{res.coverage - res.best_single:+.2f}")
    cols[3].metric("Ceiling", f"{res.ceiling:.2f}",
                   help="Coverage if each variable could use its own best site.")
    st.caption("Coverage: weighted share of the target's conditions that at least one site of the set reproduces "
               "(1 = every condition matched perfectly by some site). "
               f"{len(pool)} candidate sites (best overall plus best of each weighted dimension); "
               f"{res.method} search. " + excluded_note(ctx, t))

    missing = report.uncovered_variables(res.uncovered, w)
    if missing:
        st.warning(f"Not covered (best similarity below {config.COVERED_THRESHOLD}): {', '.join(missing)}. "
                   "No site of this set reproduces these conditions.")
    if not res.feasible:
        st.info("No set of this size satisfies the minimum-coverage requirement; showing the best unconstrained set.")

    show_table(ensemble_rows(ctx, tiles, res))
    site_names = [f"Site {chr(65 + i)} {site_label(ctx, int(tile))}" for i, tile in enumerate(tiles)]
    if charts.HAS_PLOTLY:
        left, right = st.columns(2)
        with left:
            st.subheader("Each site and what the set covers")
            show_figure(charts.ensemble_radar(sims[list(res.indices)], site_names, VAR_LABELS))
        with right:
            st.subheader("Coverage versus number of sites")
            show_figure(charts.coverage_curve([1, 2, 3], [r.coverage for r in curve], res.ceiling))
        show_figure(charts.coverage_heatmap(sims[list(res.indices)], site_names, VAR_LABELS))
        scores = np.exp(rk.log_score[pool])          # same weights and tolerances as the ensemble
        show_figure(charts.sites_map(ds.tile_lat[pool], ds.tile_lon[pool], scores,
                                     highlight=[int(i) for i in res.indices], known=known_layer(ctx),
                                     places=ctx.place[pool]))
    else:
        st.dataframe(pd.DataFrame(sims[list(res.indices)], columns=VAR_LABELS, index=site_names))


def page_limits(ctx: Context, c: dict) -> None:
    st.title("Limits Report & Site Sheet")
    st.write("An honest evaluation of where the analogy holds, where it breaks, and what Earth cannot reproduce.")
    data_banner(ctx)
    ctx, t, w, rk = compute(c)
    context_line(ctx, c, t)
    ds = ctx.ds
    target_name, body = target_info(ctx, t)

    st.subheader(f"Structural limits (Earth versus {report.body_phrase(body)})")
    show_table(report.structural_limits_table(body))
    st.subheader("What Earth cannot reproduce for this target")
    unreachable = report.reachability_table(get_reachability(t), ds.target_quantiles[t], ds.quantiles,
                                            ds.tile_lat, ds.tile_lon, ds.levels)
    if len(unreachable):
        st.caption(f"Even the closest Earth tile is farther than {config.REACHABLE_MAX_PCT:.0%} of the gaps between "
                   "two random places on Earth. These variables are excluded from the score (as MESS / ExDet do for "
                   "novel conditions) and reported here.")
        show_table(unreachable)
    else:
        st.success("Every measured variable of this target is within Earth's range.")
    st.subheader("Not evaluated in this version")
    st.markdown("\n".join(f"* {item}" for item in config.NOT_EVALUATED))
    subsurface_section(ctx, t, rk)

    sites = rk.site_idx[:config.TOP_SITES_SHOWN]
    if sites.size == 0:
        st.warning("No candidate site for the current settings.")
        return
    c["site_rank"] = int(np.clip(c["site_rank"], 1, len(sites)))
    pick = st.selectbox("Site", list(range(len(sites))), index=c["site_rank"] - 1,
                        format_func=lambda i: site_label(ctx, int(sites[i]), i + 1))
    c["site_rank"] = pick + 1
    tile = int(sites[pick])
    closer = me.closer_than_share(get_w1(t)[tile], ctx.pair_gaps)
    profile = report.profile_table(ds.quantiles[tile], ds.target_quantiles[t], rk.log_sim[tile], ds.levels, closer)
    statements = report.limits_statements(rk.log_sim[tile], profile, closer)

    st.subheader("Where the analogy holds and where it breaks")
    st.markdown("\n".join(f"* {s}" for s in statements))

    st.subheader("Site sheet")
    robustness = get_robustness(t, weights_key(w), c["tol_scale"]) if c["robustness"] else None
    lo, hi = get_score_interval(t, weights_key(w), c["tol_scale"], (tile,))[0]
    top_pct = 100 * analysis.top_share(ctx, rk.log_score, np.array([tile]))[0]
    nearest = ds.known_analogs["name"].iloc[int(ctx.known_idx[tile])]
    rel, rel_lo, rel_hi = np.exp(rk.log_score[tile] - rk.log_score.max()), lo, hi
    pct = 100 * earth_pct(ctx, c, w, rk.log_score[[tile]])[0]
    metrics = {
        "Relative score (1 = best Earth tile)": f"{rel:.2f} (90% range {rel_lo:.2f} - {rel_hi:.2f})",
        "Closer to the target than": f"{pct:.1f} % of random pairs of Earth places",
        "Top % of Earth's land": f"{top_pct:.2f}",
        "Novelty": f"{ctx.novelty[tile]:.2f}",
        "Robustness": f"{robustness[tile]:.2f}" if robustness is not None else "not computed",
        "Nearest catalogued analog": f"{nearest} ({ctx.known_dist_km[tile]:.0f} km)",
    }
    cols = st.columns(4)
    cols[0].metric("Relative score", f"{rel:.2f}", help=f"1 = best Earth tile. 90% range under weight "
                                                       f"perturbation: {rel_lo:.2f} - {rel_hi:.2f}")
    cols[1].metric("Closer than", f"{pct:.0f} % of Earth pairs",
                   help="Share of random pairs of Earth places that are less similar to each other than this site "
                        "is to the target (50 % = as close as two typical places on Earth).")
    cols[2].metric("Top % of Earth's land", f"{top_pct:.2f}",
                   help="Share of Earth's land area that scores at least as well as this site.")
    cols[3].metric("Robustness", metrics["Robustness"] if robustness is not None else "n/a",
                   help="Probability that the site stays in the best 1 % of Earth's land when the mission weights "
                        f"are perturbed and one variable is sometimes left out (Monte-Carlo, {config.ROBUSTNESS_DRAWS} "
                        "draws).")
    st.write(f"Relative score range under weight perturbation: **{rel_lo:.2f} - {rel_hi:.2f}**. "
             f"Nearest catalogued analog: "
             f"**{nearest}** ({ctx.known_dist_km[tile]:.0f} km). Data source: **{ds.meta.get('source', 'unknown')}**. "
             f"{ds.meta.get('resolution_note', '')}")

    st.subheader("What could be tested here?")
    plan = report.site_test_plan(rk.log_sim[tile], current_dimension_weights(c), body, closer,
                                 report.target_ice_note(c["target_id"]))
    left, right = st.columns(2)
    for col, sections in ((left, report.PLAN_SECTIONS[:2]), (right, report.PLAN_SECTIONS[2:])):
        with col:
            for section in sections:
                st.markdown(f"**{section}**")
                st.markdown("\n".join(f"* {i}" for i in plan[section]) or "* None")

    st.subheader("When to go? (lighting only)")
    calendar = analysis.lighting_calendar(ctx, t, tile, c["tol_scale"])
    when = report.lighting_calendar_sentence(calendar)
    if when is None:
        st.write("No Sun-elevation profile for this target.")
    else:
        st.write(when)
        show_table(report.lighting_calendar_table(calendar),
                   column_config={m: st.column_config.NumberColumn(m, format="%.2f") for m in data_prep.MONTHS})
        st.caption("Month by month, the height of the Sun above the horizon at the site (astronomy: latitude and "
                   "season) is compared with the target's over its whole year, with the tolerance of the ranking "
                   "(1 = same lighting geometry; empty = the Sun does not rise that month). "
                   "This is the lighting geometry only (shadow length, glare, tilt of solar arrays): temperature and "
                   "weather are yearly figures in this version, so they do not enter this calendar.")
        metrics["When to go (lighting only)"] = when

    label = site_label(ctx, tile, pick + 1)
    metrics["NASA Worldview"] = report.worldview_url(ds.tile_lat[tile], ds.tile_lon[tile])
    tz, zoom = zoom_view(ctx, c, t, w, rk.log_sim[tile], tile)
    if zoom is not None and zoom.order.size:
        south, north, west, east = tz.bounds()[int(zoom.order[0])]
        metrics["Best sub-area (zoom, about 11 km)"] = (
            f"centre ({(south + north) / 2:.2f}, {(west + east) / 2:.2f}), topographic similarity "
            f"{zoom.similarity[int(zoom.order[0])]:.2f} against {zoom.tile_similarity:.2f} for the whole site")
    sheet_meta = {**ds.meta, "this_target_sources": ds.meta.get("target_sources", {}).get(c["target_id"], {})}
    sheet = report.site_sheet_markdown(target_name, body, mission_name(c), label, metrics, profile, statements,
                                       plan, sheet_meta, unreachable)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download this site sheet (Markdown)", sheet.encode("utf-8"),
                       file_name=f"site_sheet_{c['target_id']}_{pick + 1}.md", mime="text/markdown")
    export = report.candidate_table(
        rk.site_idx, ds.tile_lat, ds.tile_lon, rk.log_score, ctx.novelty, ds.known_analogs["name"].tolist(),
        ctx.known_idx, ctx.known_dist_km, analysis.top_share(ctx, rk.log_score, rk.site_idx), robustness,
        earth_pct=earth_pct(ctx, c, w, rk.log_score[rk.site_idx]))
    d2.download_button("Download candidate sites (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name="candidate_sites.csv", mime="text/csv")
    d3.download_button("Download the field atlas, all targets (Markdown)",
                       get_atlas_text(tuple(sorted(current_dimension_weights(c).items())), mission_name(c)),
                       file_name="analog_atlas.md", mime="text/markdown",
                       help="Written summary for every target with the current mission weights and the default "
                            "advanced parameters; generated by rules.")


def page_atlas(ctx: Context, c: dict) -> None:
    st.title("Earth Atlas: which world does a place resemble?")
    st.write("The reverse question: start from the Earth instead of a target. Every land tile is compared with "
             f"the {len(ctx.ds.targets)} targets and takes the colour of the one it resembles most.")
    data_banner(ctx)
    context_line(ctx, c)
    ds = ctx.ds
    dim_w = current_dimension_weights(c)
    atlas = get_atlas(tuple(sorted(dim_w.items())), float(c["tol_scale"]))
    names, bodies = ds.targets["name"].tolist(), ds.targets["body"].tolist()
    best_pct = atlas.percentile.max(axis=1)

    c["atlas_min"] = st.slider("Show the places closer to a target than this share of random Earth pairs (%)",
                               30, 99, int(c.get("atlas_min", 100 * config.STATUS_PARTIAL_CLOSER)),
                               help="50 % = as close to the target as two typical places on Earth are to each other.")
    shown = best_pct >= c["atlas_min"] / 100
    if c.get("atlas_place") is None:      # default place: the tile closest to any target
        top = int(best_pct.argmax())
        c["atlas_place"] = (float(ds.tile_lat[top]), float(ds.tile_lon[top]))
    st.subheader("Look up a place")
    left, right = st.columns(2)
    lat = left.number_input("Latitude (deg)", -90.0, 90.0, float(c["atlas_place"][0]), 0.5)
    lon = right.number_input("Longitude (deg, east positive)", -180.0, 180.0, float(c["atlas_place"][1]), 0.5)
    c["atlas_place"] = (lat, lon)

    st.write(f"**{int(shown.sum()):,}** of {ds.n_tiles:,} land tiles pass the threshold: "
             f"**{100 * ctx.area[shown].sum():.1f} %** of Earth's land area; the best one reaches "
             f"{100 * best_pct.max():.0f} %.")
    for sentence in report.twin_target_sentences(names, atlas.top_overlap):
        st.caption(sentence)
    if charts.HAS_PLOTLY:
        show_figure(charts.atlas_map(ds.tile_lat[shown], ds.tile_lon[shown], atlas.best[shown], best_pct[shown],
                                     names, selected=(lat, lon), places=ctx.place[shown]))
    tiles, dist = me.nearest_tiles(ctx.tree, np.array([lat]), np.array([lon]), config.CREDIBILITY_MAX_KM)
    tile = int(tiles[0])
    if tile < 0:
        st.warning(f"No land tile within {config.CREDIBILITY_MAX_KM:.0f} km of this point (open sea, or less than "
                   "a quarter of land).")
        return
    sigma = analysis.tolerances(ctx, c["tol_scale"])
    n_targets = len(names)
    log_sim = np.stack([me.log_similarity(get_w1(t)[tile][None, :], sigma)[0] for t in range(n_targets)])
    closer = np.stack([me.closer_than_share(get_w1(t)[tile], ctx.pair_gaps) for t in range(n_targets)])
    table = report.reverse_table(names, bodies, atlas.percentile[tile], atlas.top_frac[tile], log_sim, closer)
    first = table.iloc[0]
    url = report.worldview_url(ds.tile_lat[tile], ds.tile_lon[tile])
    place = f"The tile at {site_label(ctx, tile)}, {dist[0]:.0f} km from this point,"
    if first[report.CLOSER_COLUMN] >= 50:
        st.success(f"{place} resembles **{first['Target']}** most: closer to it than "
                   f"{first[report.CLOSER_COLUMN]:.0f} % of random pairs of Earth places (top "
                   f"{first[report.RANK_COLUMN]:.2f} % of Earth's land for that target). "
                   f"[See it in NASA Worldview]({url})")
    else:
        st.info(f"{place} is not a useful analog of these targets: even its best match, {first['Target']}, is "
                f"farther from it than two typical places on Earth are from each other. "
                f"[See it in NASA Worldview]({url})")
    show_table(table)
    st.caption("Each target is judged on the variables Earth can match for it (the others are reported on page 5), "
               "with the weights of the current mission. Raw scores cannot be compared between targets, so the "
               "comparison uses the share of random pairs of Earth places that are less alike than the place and "
               "the target. A tile is 1 x 1 degree (about 110 km).")


# ======================================================================== main
def main() -> None:
    st.set_page_config(page_title="AnalogExplorer - NASA Space Apps", page_icon="🚀", layout="wide")
    try:
        clear_caches_on_new_version()
        ctx = get_context()
    except Exception as exc:  # dataset missing or breaking the data contract
        st.error(f"Could not load the dataset: {exc}")
        st.stop()
        return
    c = get_cfg()

    # st.navigation (not a radio button): each page has its own URL, opens at the top and works with the
    # browser's back button; the choices stay in st.session_state["cfg"].
    views = (lambda: page_home(ctx), lambda: page_target(ctx, c), lambda: page_profile(ctx, c),
             lambda: page_ensemble(ctx, c), lambda: page_limits(ctx, c), lambda: page_atlas(ctx, c))
    pages = {path: st.Page(view, title=title, url_path=path, icon=icon, default=path == "home")
             for (title, path, icon), view in zip(PAGES, views, strict=True)}
    new_session = NAV_KEY not in st.session_state
    st.session_state[NAV_KEY] = pages
    page = st.navigation(list(pages.values()))
    if new_session and page is not pages["home"]:
        # Every page has its own URL, and browsers reopen the last one (restored tab, address completion): a
        # visit then started on page 2 or later, before the presentation (bug of 15 Nov 2026). A new session
        # always starts on page 1; within a session, links and the back button work as usual.
        st.switch_page(pages["home"])
    sidebar(ctx)
    page.run()


if __name__ == "__main__":
    main()
