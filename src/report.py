"""Rule-based reporting (no LLM): profile tables, match status, limits statements, site sheet.

Pure functions returning pandas DataFrames / strings. No Streamlit here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src import analysis, config, math_engine

NOT_EVALUATED = "Not evaluated"


CLOSER_COLUMN = "Closer than % of Earth pairs"


def status_label(similarity: float, closer: float | None = None) -> str:
    """Strong / partial / weak match. NaN = no data for the target.

    With `closer` (share of random Earth pairs farther apart than the site is from the target,
    math_engine.closer_than_share) the thresholds come from the Earth reference
    (config.STATUS_*_CLOSER); otherwise from the similarity (config.STATUS_STRONG / PARTIAL).
    """
    value, strong, partial = ((closer, config.STATUS_STRONG_CLOSER, config.STATUS_PARTIAL_CLOSER)
                              if closer is not None else (similarity, config.STATUS_STRONG, config.STATUS_PARTIAL))
    if not np.isfinite(value):
        return NOT_EVALUATED
    if value >= strong:
        return "Strong match"
    if value >= partial:
        return "Partial match"
    return "Weak match"


def _dimension_closer(closer: np.ndarray) -> np.ndarray:
    """(D,) mean of the per-variable 'closer than' shares inside each dimension (NaN if none)."""
    out = []
    for dim in config.DIMENSIONS:
        vals = np.asarray(closer)[[i for i, d in enumerate(config.VAR_DIMENSIONS) if d == dim]]
        vals = vals[np.isfinite(vals)]
        out.append(vals.mean() if vals.size else np.nan)
    return np.array(out)


def median_of(quantile_vector: np.ndarray, levels: np.ndarray) -> float:
    """Median of a distribution described by quantiles."""
    return float(np.interp(0.5, levels, quantile_vector))


def _fmt(value: float, unit: str) -> str:
    if not np.isfinite(value):
        return "no data"
    text = f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"
    return text if unit == "0-1" else f"{text} {unit}".strip()  # indices in [0, 1] need no unit


def lower_first(text: str) -> str:
    """'Ground below 0 C at depth' -> 'ground below 0 C at depth': str.lower() would also lowercase the unit."""
    return text[:1].lower() + text[1:]


def property_label(item: str) -> str:
    """Readable name of a documented property: variable label, or the dimension name."""
    var = config.VARIABLE_BY_NAME.get(item)
    return lower_first(var.label) if var else item


def _discrepancy(site: str, target: str, similarity: float) -> str:
    """How a site departs from the target on one variable, from their formatted medians."""
    if site != target:
        return f"site {site} vs target {target}"
    if similarity >= 1 - 1e-6:
        return f"identical to the target ({target})"
    return f"same median as the target ({target}), but part of the site departs from it"


def body_phrase(body: str) -> str:
    """'the Moon' but 'Mars' (for sentences and headers)."""
    return "the Moon" if body == "Moon" else body


def profile_table(site_q: np.ndarray, target_q: np.ndarray, log_sim: np.ndarray,
                  levels: np.ndarray, closer: np.ndarray | None = None) -> pd.DataFrame:
    """One row per variable for a single site.

    site_q, target_q: (K, L) quantiles; log_sim: (K,) log-similarities of the site;
    closer: optional (K,) Earth-reference shares (status thresholds and an extra column).
    """
    sims = np.exp(log_sim)
    rows = []
    for k, var in enumerate(config.VARIABLES):
        site_med = median_of(site_q[k], levels)
        target_med = median_of(target_q[k], levels)
        row = {
            "Dimension": var.dimension,
            "Variable": var.label,
            "Similarity": float(sims[k]),
            "Status": status_label(float(sims[k]), None if closer is None else float(closer[k])),
            "Site (median)": _fmt(site_med, var.unit),
            "Target (median)": _fmt(target_med, var.unit),
            "Key discrepancy": _discrepancy(_fmt(site_med, var.unit), _fmt(target_med, var.unit), float(sims[k])),
        }
        if closer is not None:
            row[CLOSER_COLUMN] = round(100 * float(closer[k]), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def dimension_summary(log_sim: np.ndarray, closer: np.ndarray | None = None) -> pd.DataFrame:
    """Similarity per dimension (geometric mean of its variables) with its status.

    With `closer`, the status uses the mean Earth-reference share of the dimension's variables.
    """
    dim_log = math_engine.dimension_log_similarity(log_sim, config.VAR_DIMENSIONS, config.DIMENSIONS)
    sims = np.exp(dim_log)
    df = pd.DataFrame({"Dimension": list(config.DIMENSIONS), "Similarity": sims})
    if closer is None:
        df["Status"] = [status_label(float(s)) for s in sims]
    else:
        dim_closer = _dimension_closer(closer)
        df["Status"] = [status_label(float(s), float(c)) for s, c in zip(sims, dim_closer, strict=True)]
        df[CLOSER_COLUMN] = np.round(100 * dim_closer, 1)
    return df


def limits_statements(log_sim: np.ndarray, profile: pd.DataFrame, closer: np.ndarray | None = None) -> list[str]:
    """One sentence per dimension saying how far the analogy holds, and where it breaks."""
    summary = dimension_summary(log_sim, closer)
    statements = []
    for _, row in summary.iterrows():
        if row["Status"] == NOT_EVALUATED:
            statements.append(f"{row['Dimension']}: not evaluated, no data for this target.")
            continue
        sub = profile[(profile["Dimension"] == row["Dimension"]) & profile["Similarity"].notna()]
        worst = sub.loc[sub["Similarity"].idxmin()]
        verb = {"Strong match": "holds well", "Partial match": "holds partially",
                "Weak match": "breaks down"}[row["Status"]]
        reference = f", closer than {row[CLOSER_COLUMN]:.0f} % of Earth pairs" if CLOSER_COLUMN in row else ""
        statements.append(
            f"{row['Dimension']}: the analogy {verb} (s = {row['Similarity']:.2f}{reference}). "
            f"Largest gap: {lower_first(worst['Variable'])}, {worst['Key discrepancy']}.")
    return statements


REACH_COLUMNS = ("Variable", "Target (median)", "Closest Earth tile (median)", "Closest tile at",
                 "Gap (x sigma)", "Farther than % of Earth-Earth gaps")


def reachability_table(reach, target_q: np.ndarray, tile_q: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                       levels: np.ndarray) -> pd.DataFrame:
    """Variables out of Earth's range for the target (math_engine.Reachability), with figures.

    target_q: (K, L) target quantiles; tile_q: (n, K, L) Earth tile quantiles.
    """
    rows = []
    for k, var in enumerate(config.VARIABLES):
        if reach.reachable[k]:
            continue
        t = int(reach.best_tile[k])
        rows.append(dict(zip(REACH_COLUMNS, (
            var.label, _fmt(median_of(target_q[k], levels), var.unit),
            _fmt(median_of(tile_q[t, k], levels), var.unit), f"({lat[t]:.1f}, {lon[t]:.1f})",
            round(float(reach.gap_sigma[k]), 1), round(100 * float(reach.percentile[k]), 2)), strict=True)))
    return pd.DataFrame(rows, columns=list(REACH_COLUMNS))


def structural_limits_table(body: str | None = None) -> pd.DataFrame:
    """Structural limits of Earth; restricted to the target body when it is known."""
    df = pd.DataFrame(config.STRUCTURAL_LIMITS)
    if body in df.columns:
        return df[["Factor", body, "Impact"]].rename(columns={body: f"On {body_phrase(body)}"})
    return df


# ======================================================== site sheet (5.5)
PLAN_SECTIONS = ("Representative tests", "Possible with caution", "Not representative here",
                 "Never transferable from Earth")


def target_ice_note(target_id: str) -> str | None:
    """Why ice prospecting does not apply to a target (config.TARGET_NEAR_SURFACE_ICE), None when ice is documented
    near its surface or the target is unknown."""
    documented, source = config.TARGET_NEAR_SURFACE_ICE.get(str(target_id), (True, ""))
    return None if documented else source


def site_test_plan(log_sim: np.ndarray, dim_weights: Mapping[str, float], body: str,
                   closer: np.ndarray | None = None, no_ice_reason: str | None = None) -> dict[str, list[str]]:
    """What the site can be used for, given its per-dimension match and the mission.

    Only the dimensions the mission cares about (weight > 0) are considered; if none,
    all of them. Structural limits of the target body are never transferable.
    closer: optional Earth-reference shares used for the statuses (see dimension_summary).
    no_ice_reason: set when no ice is documented near the target's surface (target_ice_note): ice prospecting is
    then never proposed, since a match on ground below 0 C is a temperature match, not an ice match.
    """
    summary = dimension_summary(log_sim, closer)
    relevant = [d for d in config.DIMENSIONS if dim_weights.get(d, 0.0) > 0] or list(config.DIMENSIONS)
    plan: dict[str, list[str]] = {s: [] for s in PLAN_SECTIONS}
    for _, row in summary[summary["Dimension"].isin(relevant)].iterrows():
        dim, s = row["Dimension"], float(row["Similarity"])
        activities = config.TEST_ACTIVITIES.get(dim, ())
        if no_ice_reason and config.ICE_PROSPECTING_ACTIVITY in activities:
            activities = tuple(a for a in activities if a != config.ICE_PROSPECTING_ACTIVITY)
            plan["Not representative here"].append(
                f"Ice prospecting: no ice is documented near the surface of this target ({no_ice_reason.rstrip('.')}); "
                "the water / ice match is a match on dryness and ground temperature only.")
        if row["Status"] == "Strong match":
            plan["Representative tests"].extend(activities)
        elif row["Status"] == "Partial match":
            plan["Possible with caution"].extend(
                f"{a} (partial analogy on {dim.lower()}, s = {s:.2f})" for a in activities)
        elif row["Status"] == NOT_EVALUATED:
            plan["Not representative here"].append(
                f"{dim} (not evaluated): no target data, so this comparison cannot support these tests.")
        else:
            plan["Not representative here"].append(
                f"{dim} (s = {s:.2f}): tests that depend on it would give misleading results.")
    for row in config.STRUCTURAL_LIMITS:
        if body in row:
            plan["Never transferable from Earth"].append(f"{row['Factor']} ({row[body]}): {row['Impact']}")
    return plan


def markdown_table(df: pd.DataFrame) -> str:
    """Minimal Markdown table (no extra dependency)."""
    def cell(v) -> str:
        if isinstance(v, (float, np.floating)):
            return f"{v:.2f}" if np.isfinite(v) else "n/a"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def site_sheet_markdown(target_name: str, body: str, mission: str, site_label: str,
                        metrics: Mapping[str, str], profile: pd.DataFrame, statements: list[str],
                        plan: Mapping[str, list[str]], meta: Mapping,
                        unreachable: pd.DataFrame | None = None) -> str:
    """Complete site sheet as Markdown: profile, limits, test plan, sources and uncertainties.

    unreachable: optional reachability_table of the target (conditions Earth cannot reproduce).
    """
    out = [f"# Site sheet: {site_label}", "",
           f"**Target:** {target_name} ({body})  ", f"**Mission:** {mission}", ""]
    out += ["## Key figures", ""] + [f"- **{k}:** {v}" for k, v in metrics.items()] + [""]
    out += ["## Profile by variable", "",
            markdown_table(profile[["Dimension", "Variable", "Similarity", "Status", "Site (median)",
                               "Target (median)"]]), ""]
    out += ["## Where the analogy holds and where it breaks", ""] + [f"- {s}" for s in statements] + [""]
    if unreachable is not None and len(unreachable):
        out += ["## What Earth cannot reproduce for this target", "",
                "Even the closest Earth tile is farther than 99 % of the gaps between two random places on Earth; "
                "these variables are excluded from the score and reported here.", "", markdown_table(unreachable), ""]
    out += ["## What to test here", ""]
    for section in PLAN_SECTIONS:
        items = plan.get(section, [])
        out += [f"### {section}", ""] + ([f"- {i}" for i in items] or ["- None"]) + [""]
    out += ["## Not evaluated in this version", ""] + [f"- {d}" for d in config.NOT_EVALUATED] + [""]
    out += ["## Sources and uncertainties", "",
            f"- Data source: {meta.get('source', 'unknown')} (version {meta.get('version', '?')})"]
    out += [f"- {k}: {v}" for k, v in meta.get("sources", {}).items()]
    out += [f"- Target {lower_first(config.VARIABLE_BY_NAME[k].label)}: {v}"
            for k, v in meta.get("this_target_sources", {}).items() if k in config.VARIABLE_BY_NAME]
    out += [
            f"- {meta.get('resolution_note') or 'Resolutions: to be documented with the real datasets.'}",
            f"- Status: strong when the site is closer to the target than {config.STATUS_STRONG_CLOSER:.0%} of random "
            f"pairs of Earth places, partial above {config.STATUS_PARTIAL_CLOSER:.0%}. Thresholds and mission weights "
            "are hypotheses to calibrate.",
            "- Scores are relative to the best Earth tile; variables out of Earth's range are excluded from them.",
            "- Shares of Earth ('closer than X % of Earth pairs', 'top X %') are shares of land area: a tile counts "
            "for its land surface.",
            "- Robustness: Monte-Carlo perturbation of the mission weights (Dirichlet).", ""]
    return "\n".join(out)


# ========================================================= fine-scale zoom (3.6)
def zoom_table(bounds: np.ndarray, sub_quantiles: np.ndarray, similarity: np.ndarray, order: np.ndarray,
               levels: np.ndarray, top: int = 3) -> pd.DataFrame:
    """Best sub-areas of a site: centre, size, median slope and roughness, topographic similarity.

    bounds: (S, 4) south, north, west, east in degrees; sub_quantiles: (S, 2, L) slope and
    roughness; similarity: (S,); order: sub-areas with enough land, best first.
    """
    rows = []
    for rank, i in enumerate(np.asarray(order)[:top], start=1):
        south, north, west, east = bounds[i]
        lat, lon = 0.5 * (south + north), 0.5 * (west + east)
        height = math_engine.haversine_km(south, lon, north, lon)
        width = math_engine.haversine_km(lat, west, lat, east)
        rows.append({
            "Rank": rank, "Lat": round(float(lat), 3), "Lon": round(float(lon), 3),
            "Size (km)": f"{float(height):.0f} x {float(width):.0f}",
            "Slope (median)": _fmt(median_of(sub_quantiles[i, 0], levels), "deg"),
            "Roughness (median)": _fmt(median_of(sub_quantiles[i, 1], levels), "m"),
            "Topographic similarity": float(similarity[i]),
            "NASA Worldview": worldview_url(lat, lon, half_deg=max(float(north - south), float(east - west))),
        })
    return pd.DataFrame(rows)


# ================================================= subsurface case studies (3.1)
SUBSURFACE_GAP = "Gap to the target's interior (K)"


def subsurface_table(sites: pd.DataFrame, target_temp_k: float, top_frac: np.ndarray | None = None,
                     understudied: np.ndarray | None = None) -> pd.DataFrame:
    """Lava-tube case studies, closest expected cave temperature to the target's first.

    sites: the dataset's subsurface table (maat_k = mean annual air temperature of the site's
    climate cell, the estimate of the temperature inside the tube). top_frac: optional (n,)
    rank of the tile around each site for the current target and mission (NaN = no tile);
    understudied: optional (n,) bool.
    """
    gap = sites["maat_k"].to_numpy(dtype=float) - float(target_temp_k)
    data = {
        "Lava-tube field": sites["name"].to_numpy(), "Country": sites["country"].to_numpy(),
        "Lat": sites["lat"].round(2).to_numpy(), "Lon": sites["lon"].round(2).to_numpy(),
        "Mean annual air temperature (K)": np.round(sites["maat_k"].to_numpy(dtype=float), 1),
        SUBSURFACE_GAP: np.round(gap, 1),
    }
    if top_frac is not None:
        data["Surface match (top % of Earth tiles)"] = np.round(100 * np.asarray(top_frac, dtype=float), 1)
    if understudied is not None:
        data["Under-studied"] = np.asarray(understudied, dtype=bool)
    data["NASA Worldview"] = [worldview_url(lat, lon) for lat, lon in zip(sites["lat"], sites["lon"], strict=True)]
    order = np.argsort(np.abs(gap), kind="stable")
    return pd.DataFrame(data).iloc[order].reset_index(drop=True)


# ====================================================== reverse search (atlas)
RANK_COLUMN = "Top % of Earth's land"


def reverse_table(target_names: list[str], bodies: list[str], percentile: np.ndarray, top_frac: np.ndarray,
                  log_sim: np.ndarray, closer: np.ndarray) -> pd.DataFrame:
    """One Earth place against every target, best match first.

    percentile, top_frac: (T,) absolute rank of the place for each target and share of Earth's land that
    scores at least as well;
    log_sim, closer: (T, K) per-variable log-similarities and Earth-reference shares.
    The strongest and weakest dimensions are those with the highest and lowest Earth-reference share.
    """
    def describe(row) -> str:
        return f"{row['Dimension']} ({row['Status'].lower()})"

    rows = []
    for t, (name, body) in enumerate(zip(target_names, bodies, strict=True)):
        summary = dimension_summary(log_sim[t], closer[t])
        evaluated = summary[summary["Status"] != NOT_EVALUATED]
        rows.append({
            "Target": name, "Body": body,
            CLOSER_COLUMN: round(100 * float(percentile[t]), 1),
            RANK_COLUMN: round(100 * float(top_frac[t]), 2),
            "Strongest dimension": (describe(evaluated.loc[evaluated[CLOSER_COLUMN].idxmax()])
                                    if len(evaluated) else "-"),
            "Weakest dimension": (describe(evaluated.loc[evaluated[CLOSER_COLUMN].idxmin()])
                                  if len(evaluated) else "-"),
        })
    return pd.DataFrame(rows).sort_values(CLOSER_COLUMN, ascending=False).reset_index(drop=True)


def uncovered_variables(uncovered: np.ndarray, weights: np.ndarray) -> list[str]:
    """Labels of the variables of the score that a complementary set does not cover (math_engine.EnsembleResult.
    uncovered). A variable without weight is not listed: the mission ignores it or Earth cannot match it (it is
    then in the limits report), and the set was not chosen for it."""
    return [v.label for v, u, w in zip(config.VARIABLES, uncovered, weights, strict=True) if u and w > 0]


def weak_best_site_note(best_pct: float) -> str | None:
    """Sentence said when even the best Earth site is farther from the target than two typical places on Earth
    are from each other (below config.STATUS_PARTIAL_CLOSER): the ranking then orders poor analogs. None otherwise.

    best_pct: share in [0, 1] of random pairs of Earth places less alike than the best site and the target.
    """
    if not np.isfinite(best_pct) or best_pct >= config.STATUS_PARTIAL_CLOSER:
        return None
    return (f"No single place on Earth is a useful overall analog of this target for this mission: even the best "
            f"site is closer to it than only {100 * best_pct:.0f} % of random pairs of Earth places (50 % = as alike "
            "as two ordinary places on Earth). Read the profile by dimension and the complementary set instead of "
            "the rank.")


# ======================================================== near-duplicate targets
def twin_target_sentences(target_names: list[str], top_overlap: np.ndarray,
                          only: int | None = None) -> list[str]:
    """One sentence per pair of targets whose best land on Earth is mostly the same places.

    top_overlap: (T, T) overlaps (analysis.Atlas.top_overlap); only: a target index, to get a single sentence
    naming the targets that share its best places (empty list when there is none).
    """
    twins = analysis.twin_targets(top_overlap)
    share = f"best {config.ROBUSTNESS_TOP_FRAC:.0%} of Earth's land"
    why = ("what weighs in this mission is alike for them at the scale of a tile of about 110 km, so read them as "
           "windows on one case, not as independent results.")
    if only is not None:
        others = [(j if i == only else i, overlap) for i, j, overlap in twins if only in (i, j)]
        if not others:
            return []
        listed = ", ".join(f"{target_names[k]} ({100 * overlap:.0f} %)" for k, overlap in others)
        return [f"For this mission, {target_names[only]} points to mostly the same places on Earth as: {listed} "
                f"(overlap of their {share}); {why}"]
    return [f"{target_names[i]} and {target_names[j]} point to mostly the same places for this mission: their "
            f"{share} overlap by {100 * overlap:.0f} %; {why}" for i, j, overlap in twins]


def p_text(p: float) -> str:
    """'p = 0.041', or 'p < 0.001' rather than a misleading 'p = 0.000'."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ======================================================== when to go (5.5)
def lighting_calendar_table(similarity: np.ndarray) -> pd.DataFrame:
    """One row, one column per month: how close the Sun's elevation at the site is to the target's
    (analysis.lighting_calendar); empty for a month when the Sun never rises."""
    from src.data_prep import MONTHS

    return pd.DataFrame([np.round(np.asarray(similarity, dtype=float), 2)], columns=list(MONTHS))


def lighting_calendar_sentence(similarity: np.ndarray, top: int = 3) -> str | None:
    """'Lighting closest to the target in Mar, Sep, Oct (similarity 0.71 to 0.80)'; None without any profile."""
    from src.data_prep import MONTHS

    sims = np.asarray(similarity, dtype=float)
    lit = np.flatnonzero(np.isfinite(sims))
    if lit.size == 0:
        return None
    best = np.sort(lit[np.argsort(-sims[lit], kind="stable")][:top])
    text = (f"The Sun stands most like on the target in {', '.join(MONTHS[m] for m in best)} (similarity of its "
            f"elevation {sims[best].min():.2f} to {sims[best].max():.2f}; over the whole year the months range from "
            f"{sims[lit].min():.2f} to {sims[lit].max():.2f})")
    dark = 12 - lit.size
    return text + (f"; the Sun does not rise during {dark} months." if dark else ".")


# ================================================================ data chain
def data_chain_table() -> pd.DataFrame:
    """From open data to the score: one row per variable with its dimension and its source on each world
    (config.VARIABLE_SOURCES)."""
    rows = []
    for var in config.VARIABLES:
        earth, moon, mars = config.VARIABLE_SOURCES[var.name]
        rows.append({"Dimension": var.dimension, "Variable": f"{var.label} ({var.unit})",
                     "Earth (every land tile)": earth, "Moon targets": moon, "Mars targets": mars})
    return pd.DataFrame(rows)


# ============================================================ challenge map
CHALLENGE_COLUMNS = ("Type", "Challenge text", "Status", "What this version does", "Page")


def challenge_table() -> pd.DataFrame:
    """The official challenge text, item by item, with what this version does about it (config.CHALLENGE_MAP)."""
    return pd.DataFrame(list(config.CHALLENGE_MAP), columns=list(CHALLENGE_COLUMNS))


def engine_steps() -> list[tuple[str, str, str]]:
    """(title, Markdown detail with LaTeX formulas, page where it shows) of each step of the engine, from the data
    to the ranking, for the page 1 diagram. Thresholds are read from config, so the
    diagram follows the code."""
    huber = ("" if config.HUBER_DELTA is None
             else f", then linear beyond $z = {config.HUBER_DELTA:g}$ (Huber)")
    return [
        ("Open data", "Earth: NASA POWER, Copernicus DEM, ESA WorldCover, SCAR. Moon and Mars: NASA PDS (LRO, MOLA), "
                      "NASA PGDA, Mars Climate Database.", "full list: Data sources"),
        ("Distributions", f"Each 1° x 1° land tile and each target is described by the quantiles $q$ of "
                          f"{len(config.VARIABLES)} variables in {len(config.DIMENSIONS)} dimensions.",
         "page 3, Distributions"),
        ("Gap per variable", r"Wasserstein distance: $W_1 = \overline{|\,q_{site} - q_{target}\,|}$, the mean gap "
                             "between the quantile curves.", "shaded area, page 3"),
        ("Can Earth match it?", f"A variable whose closest Earth tile is farther than {config.REACHABLE_MAX_PCT:.0%} "
                                "of the gaps between two Earth places leaves the score and is reported.",
         "pages 2 and 5"),
        ("Similarity", r"$z = W_1 / \sigma$, with $\sigma$ the typical gap between two Earth places; "
                       r"$\log s = -z^2/2$" + huber + ".", "page 3, Profile"),
        ("Mission weights", r"Each dimension weight is spread over its variables: $\sum_k w_k = 1$.", "page 2"),
        ("Score", r"$S = \exp\left(\sum_k w_k \log s_k\right)$, a weighted geometric mean: a weak variable is "
                  "not compensated by the others.", "page 3"),
        ("Distinct sites", f"Best sites at least {config.DEFAULT_RADIUS_KM:.0f} km apart (adjustable); optional "
                           rf"novelty $N = 1 - e^{{-d / {config.NOVELTY_LAMBDA_KM:.0f}\,\mathrm{{km}}}}$, $d$ = "
                           "distance to the nearest known analog.", "page 3"),
    ]


ENGINE_OUTPUTS = (
    ("Relative score, closer than % of Earth pairs, top % of land", "page 3"),
    (f"Robustness: Monte-Carlo on the weights ({config.ROBUSTNESS_DRAWS} draws)", "page 3"),
    ("Complementary set of 1 to 3 sites", "page 4"),
    ("Credibility check against known analogs", "page 3"),
    ("Limits report and site sheet", "page 5"),
)


def challenge_counts() -> dict[str, int]:
    """Number of challenge items per status, in the order of config.CHALLENGE_STATUSES."""
    statuses = [row[2] for row in config.CHALLENGE_MAP]
    return {s: statuses.count(s) for s in config.CHALLENGE_STATUSES}


# ================================================================ candidates
PLACE_COLUMN = "Place"


def site_name(place: str | None, lat: float, lon: float, digits: int = 1) -> str:
    """'Kunlun Mountains, China (36.5 N, 86.5 E)', or the coordinates alone when the place has no name."""
    where = coords_text(lat, lon, digits)
    return f"{place} ({where})" if place else where


def worldview_url(lat: float, lon: float, half_deg: float = 0.75) -> str:
    """NASA Worldview view of an Earth site (satellite imagery around the tile)."""
    return (f"https://worldview.earthdata.nasa.gov/?v={lon - half_deg:.2f},{lat - half_deg:.2f},"
            f"{lon + half_deg:.2f},{lat + half_deg:.2f}")


def candidate_table(site_idx: np.ndarray, lat: np.ndarray, lon: np.ndarray, log_score_: np.ndarray,
                    novelty_: np.ndarray, known_names: list[str], known_idx: np.ndarray,
                    known_km: np.ndarray, top_frac: np.ndarray,
                    robustness_: np.ndarray | None = None,
                    interval: np.ndarray | None = None,
                    earth_pct: np.ndarray | None = None,
                    understudied: np.ndarray | None = None,
                    places: np.ndarray | None = None) -> pd.DataFrame:
    """Table of distinct candidate sites ordered as given (best first).

    The score is never shown raw: relative to the best tile (1 = best)
    and, with earth_pct, as an absolute percentile against random Earth-Earth pairs.
    interval: optional (len(site_idx), 2) range of the relative score under weight perturbation
    (math_engine.score_interval with relative=True: each draw is divided by its own best tile).
    places: optional (n,) display name of every tile (analysis.Context.place), shown as a 'Place' column.
    """
    best = float(np.max(log_score_))
    data = {
        "Rank": np.arange(1, len(site_idx) + 1),
        **({} if places is None else {PLACE_COLUMN: [str(places[i]) or "-" for i in site_idx]}),
        "Tile": site_idx,
        "Lat": np.round(lat[site_idx], 2),
        "Lon": np.round(lon[site_idx], 2),
        "Relative score": np.exp(log_score_[site_idx] - best),
    }
    if interval is not None:
        data["Relative score 90% range"] = [f"{lo:.2f} - {hi:.2f}" for lo, hi in np.asarray(interval)]
    if earth_pct is not None:
        data["Closer than % of Earth pairs"] = np.round(100 * np.asarray(earth_pct), 1)
    if understudied is not None:
        data["Under-studied"] = np.asarray(understudied)[site_idx]
    data.update({
        "Top %": np.round(100 * top_frac, 2),
        "Novelty": novelty_[site_idx],
        "Nearest known analog": [known_names[i] for i in known_idx[site_idx]],
        "Distance (km)": np.round(known_km[site_idx]),
    })
    if robustness_ is not None:
        data["Robustness"] = robustness_[site_idx]
    data["NASA Worldview"] = [worldview_url(lat[i], lon[i]) for i in site_idx]
    return pd.DataFrame(data)


# ======================================================== written atlas (5.5)
ATLAS_DEFAULT_MISSION = "Habitat / Construction"   # weighs the four dimensions, closest to a permanent base
ATLAS_METHOD = (
    "Earth's land is cut into tiles of 1 x 1 degree (about 110 km). Each tile and each target is described, "
    "for every variable, by the distribution of its values, not by an average.",
    "A tile is compared with a target variable by variable (distance between the two distributions). The gap is "
    "judged against a yardstick: how different two places on Earth picked at random are from each other.",
    "The score is a weighted geometric mean, so a very poor match on one variable cannot be hidden by good "
    "matches elsewhere. The weights come from the mission profile. The penalty grows with the square of the gap "
    "for close matches and linearly for distant ones, so one very poor variable does not decide the order alone.",
    "A variable that no place on Earth can match (for example the lunar day/night temperature swing) leaves the "
    "score and is reported with figures instead of quietly deciding the ranking.",
    "Neighbouring tiles look alike, so the list keeps distinct sites only (at least 300 km apart).",
    "Under-studied means at least 500 km from every analog of our sourced catalogue and outside Antarctica, the "
    "Atacama, Iceland and Hawaii, which the challenge names as already well studied.",
    "Shares of Earth are shares of land area, not counts of tiles: a pair of 'random places on Earth' is two "
    "points drawn on land, and 'top 1 %' is the best 1 % of the land surface.",
    "The credibility check asks whether analog sites known from the literature rank better than random places "
    "on land on the property they are known for. Its protocol was fixed before looking at the result; it was "
    "revised once afterwards (ranks as shares of land area, tied places sharing their average rank), and both "
    "sets of results are kept by the team. " + config.CREDIBILITY_CAVEAT,
    "Score and status use one yardstick, the tolerance: for a variable where many pairs of places are identical "
    "(bare ground, ground below 0 C), the site's gap is scaled by the share of pairs that differ, so that 50 % "
    "means 'one tolerance away' for every variable.",
    "A status reads 'closer than X % of Earth pairs' among the pairs that differ on the variable: two places that "
    "both have no bare ground at all, or are both frozen all year, say nothing about how close a match is.",
)


def coords_text(lat: float, lon: float, digits: int = 1) -> str:
    """'68.5 N, 92.5 E' style coordinates."""
    return (f"{abs(float(lat)):.{digits}f} {'N' if lat >= 0 else 'S'}, "
            f"{abs(float(lon)):.{digits}f} {'E' if lon >= 0 else 'W'}")


def small_tested_analogs(cred, known: pd.DataFrame) -> list[str]:
    """Tested known analogs smaller than a zoom sub-area (config.SMALL_ANALOG_KM): diluted in their tile.
    Empty when the catalogue gives no extent (synthetic test catalogue)."""
    if "extent_km" not in known.columns:
        return []
    return [str(n) for n, p, e in zip(known["name"], cred.props, known["extent_km"], strict=True)
            if p and float(e) < config.SMALL_ANALOG_KM]


def credibility_caveats(cred, known: pd.DataFrame, targets: pd.DataFrame, p_holm: float | None = None) -> list[str]:
    """What qualifies a credibility verdict: Holm-corrected p, cross-check with the other body, known
    analogs too small for a tile. Plain sentences, for the app and the written atlas."""
    test, out = cred.test, []
    if test.n_sites == 0:
        return out
    if p_holm is not None and np.isfinite(p_holm):
        holds = "would still pass" if p_holm < config.CREDIBILITY_TEST["alpha"] else "would not pass"
        out.append(f"After a Holm correction for testing the {len(targets)} targets, {p_text(p_holm)}: the check "
                   f"{holds}.")
    control = cred.control
    if control is not None and np.isfinite(control).any():
        other = targets.iloc[cred.control_target]
        better = np.nanmean(control) < test.mean_rank
        out.append(f"Cross-check: for {other['name']} ({body_phrase(str(other['body']))}), the same sites rank in the "
                   f"top {100 * np.nanmean(control):.0f} % (top {100 * test.mean_rank:.0f} % here). "
                   + ("They rank better there than for their own body, so the check does not show that the ranking "
                      "is specific to this target." if better else
                      "They rank better for their own body, as a method that tells worlds apart should."))
    worst = getattr(cred, "worst_var", None)
    if worst is not None:
        gaps = np.asarray(worst)[np.isfinite(cred.ranks) & ~np.asarray(cred.at_top)]
        if gaps.size:
            values, counts = np.unique(gaps, return_counts=True)
            label = lower_first(config.VARIABLES[int(values[counts.argmax()])].label)
            out.append(f"Read from the data: for {int(counts.max())} of the {int(np.isfinite(cred.ranks).sum())} "
                       f"ranked sites, the tested property farthest from the target is {label} (details in the "
                       "table).")
    small = small_tested_analogs(cred, known)
    if small:
        out.append(f"{len(small)} tested sites are smaller than {config.SMALL_ANALOG_KM:.0f} km ({', '.join(small)}) "
                   "and are diluted in a tile of about 110 km.")
    return out


WHY_COLUMN = "Why this rank"
TIE_SHARE_SHOWN = 0.05     # ties are mentioned when at least this share of the land matches the target exactly


def _rank_reason(cred, i: int, props: tuple[str, ...], extent_km: float | None, tile_q: np.ndarray,
                 target_q: np.ndarray, levels: np.ndarray) -> str:
    """Why known analog `i` has its rank, read from the data (no judgement): no tile, nothing comparable, a tie
    for first place, or the tested variable with the largest gap; plus the dilution of a small site."""
    tile = int(cred.tiles[i])
    if tile < 0:
        return (f"Not ranked: no land tile within {config.CREDIBILITY_MAX_KM:.0f} km (the nearest is "
                f"{cred.dist_km[i]:.0f} km away; small island).")
    if not np.isfinite(cred.ranks[i]):
        return ("Not ranked: none of its documented properties can be compared for this target (not measured, or "
                "out of Earth's range).")
    tie = float(cred.top_tie[i])
    if cred.at_top[i]:
        text = (f"Matches the target exactly on {', '.join(property_label(p) for p in props)}, like {tie:.0%} of "
                "Earth's land: all tied for first place, and the rank is the middle of that tie.")
    else:
        k = int(cred.worst_var[i])
        var = config.VARIABLES[k]
        site = _fmt(median_of(tile_q[tile, k], levels), var.unit)
        text = (f"Largest gap: {lower_first(var.label)} (s = {float(cred.worst_sim[i]):.2f}), "
                f"{_discrepancy(site, _fmt(median_of(target_q[k], levels), var.unit), float(cred.worst_sim[i]))}.")
        if tie >= TIE_SHARE_SHOWN:
            text += f" {tie:.0%} of Earth's land matches the target exactly on what is tested and ranks ahead."
    if extent_km is not None and extent_km < config.SMALL_ANALOG_KM:
        text += f" The site is about {extent_km:g} km wide: diluted in a tile of about 110 km."
    return text


def credibility_table(cred, known: pd.DataFrame, tile_q: np.ndarray, target_q: np.ndarray,
                      levels: np.ndarray) -> pd.DataFrame:
    """The tested known analogs with their rank and the reason for it.

    cred: analysis.Credibility; tile_q: (n, K, L) Earth tile quantiles; target_q: (K, L) of the target.
    A low rank is reported as is and explained by the data (largest gap among the tested variables, ties,
    site smaller than a tile), never corrected.
    """
    extents = known["extent_km"].tolist() if "extent_km" in known.columns else [None] * len(known)
    rows = []
    for i, (name, props) in enumerate(zip(known["name"], cred.props, strict=True)):
        if not props:
            continue
        row = {"Known analog": str(name), "Tested on": ", ".join(property_label(p) for p in props),
               "Nearest tile (km)": round(float(cred.dist_km[i])),
               "Top % (own properties)": round(100 * float(cred.ranks[i]), 1)}
        if cred.control is not None:
            row["Top % for the other body"] = round(100 * float(cred.control[i]), 1)
        extent = None if extents[i] is None else float(extents[i])
        row[WHY_COLUMN] = _rank_reason(cred, i, props, extent, tile_q, target_q, levels)
        rows.append(row)
    return pd.DataFrame(rows)


def _credibility_sentence(cred, body: str, known: pd.DataFrame | None = None, targets: pd.DataFrame | None = None,
                          p_holm: float | None = None) -> str:
    test = cred.test
    if test.n_sites == 0:
        return f"No known {body} analog of our catalogue could be tested for this target."
    verdict = ("passes" if test.passed else "fails" if test.conclusive
               else f"is inconclusive (fewer than {int(config.CREDIBILITY_TEST['min_sites'])} sites)")
    text = (f"The {test.n_sites} known {body} analogs of our sourced catalogue rank on average in the top "
            f"{100 * test.mean_rank:.0f} % of Earth's land on the property each is known for (random places on "
            f"land: top {100 * test.null_mean:.0f} %, {p_text(test.p_value)}): the check {verdict}.")
    if known is None or targets is None:
        return text
    return " ".join([text, *credibility_caveats(cred, known, targets, p_holm)])


BEST_SITE_COLUMN = "Best site (closer than % of Earth pairs)"


def headline_row(ctx, t: int, a, cred, p_holm: float | None = None, tol_scale: float = 1.0) -> dict:
    """One line of results for a target: its best Earth site, its best under-studied site, what Earth cannot
    reproduce and the verdict of the credibility check. Shared by the atlas summary and the app's overview,
    so that both show the same figures.

    a: analysis.TargetAnalysis of the target (ranking without novelty); cred: its analysis.Credibility.
    """
    ds = ctx.ds
    target = ds.targets.iloc[t]
    rk, sites = a.ranking, a.ranking.site_idx
    sigma = analysis.tolerances(ctx, tol_scale)

    def where(tile: int) -> str:
        return site_name(ctx.place[tile], ds.tile_lat[tile], ds.tile_lon[tile])

    best_text = under_text = "none"
    if sites.size:
        best_pct = float(math_engine.earth_percentile(rk.log_score[sites[:1]], ctx.pair_gaps, sigma, a.weights)[0])
        best_text = f"{where(int(sites[0]))}: {100 * best_pct:.0f} %"
    tile = analysis.best_understudied(ctx, sites)
    if tile is not None:
        under_text = f"{where(tile)}, rank {int(np.flatnonzero(sites == tile)[0]) + 1}"
    out_of_range = [lower_first(v.label) for v, ok in zip(config.VARIABLES, a.reach.reachable, strict=True) if not ok]
    return {"Target": f"{target['name']} ({target['body']})", BEST_SITE_COLUMN: best_text,
            "Best under-studied site": under_text, "Out of Earth's range": ", ".join(out_of_range) or "nothing",
            "Credibility check": ("pass" if cred.test.passed else "fail" if cred.test.conclusive else "inconclusive"),
            "After Holm correction": ("n/a" if p_holm is None or not np.isfinite(p_holm) else
                                      "pass" if p_holm < config.CREDIBILITY_TEST["alpha"] else "fail")}


def overview_table(ctx, dim_weights: Mapping[str, float], creds: Sequence, p_holm: np.ndarray,
                   tol_scale: float = 1.0, w1_by_target: Mapping[int, np.ndarray] | None = None) -> pd.DataFrame:
    """headline_row of every target for a mission (dimension weights): the results at a glance.

    creds, p_holm: analysis.credibility_all of the same tolerance scale; w1_by_target: optional cached distances.
    """
    rows = []
    for t in range(len(ctx.ds.targets)):
        a = analysis.analyse_target(ctx, t, dim_weights, tol_scale, w1=(w1_by_target or {}).get(t))
        rows.append(headline_row(ctx, t, a, creds[t], float(p_holm[t]), tol_scale))
    return pd.DataFrame(rows)


def _atlas_target(ctx, t: int, dim_weights: Mapping[str, float], top: int, cred=None,
                  p_holm: float | None = None) -> tuple[list[str], dict]:
    """Markdown lines of one target and its row of the summary table.

    cred, p_holm: the target's credibility check and its Holm-corrected p (analysis.credibility_all).
    """
    ds = ctx.ds
    target = ds.targets.iloc[t]
    name, body = str(target["name"]), str(target["body"])
    a = analysis.analyse_target(ctx, t, dim_weights)
    rk, sites = a.ranking, a.ranking.site_idx
    sigma = analysis.tolerances(ctx)
    known_names = ds.known_analogs["name"].tolist()

    def named(tile: int) -> str:
        return site_name(ctx.place[tile], ds.tile_lat[tile], ds.tile_lon[tile])

    out = [f"### {name} ({body})", "",
           f"Target window of about 110 km around {coords_text(target['lat'], target['lon'], 2)}. {target['note']}", ""]

    unreachable = reachability_table(a.reach, ds.target_quantiles[t], ds.quantiles, ds.tile_lat, ds.tile_lon, ds.levels)
    out += ["**What Earth cannot reproduce here.**", ""]
    if len(unreachable):
        out += ["Even the closest Earth tile is farther from the target than 99 % of the gaps between two random "
                "places on Earth, so these variables are left out of the score:", "", markdown_table(unreachable), ""]
    else:
        out += ["Every measured variable of this target is within Earth's range.", ""]

    shown = sites[:top]
    table = candidate_table(shown, ds.tile_lat, ds.tile_lon, rk.log_score, ctx.novelty, known_names, ctx.known_idx,
                            ctx.known_dist_km, analysis.top_share(ctx, rk.log_score, shown),
                            earth_pct=math_engine.earth_percentile(rk.log_score[shown], ctx.pair_gaps, sigma,
                                                                   a.weights),
                            understudied=ctx.understudied, places=ctx.place)
    table["NASA Worldview"] = [f"[imagery]({url})" for url in table["NASA Worldview"]]
    out += ["**Best distinct sites.**", "",
            markdown_table(table[["Rank", PLACE_COLUMN, "Lat", "Lon", "Relative score", CLOSER_COLUMN,
                                  "Under-studied", "Nearest known analog", "Distance (km)", "NASA Worldview"]]), ""]
    cred = analysis.credibility(ctx, t) if cred is None else cred
    summary = headline_row(ctx, t, a, cred, p_holm)
    if sites.size:
        weak = weak_best_site_note(float(math_engine.earth_percentile(rk.log_score[sites[:1]], ctx.pair_gaps,
                                                                      sigma, a.weights)[0]))
        out += [weak, ""] if weak else []

    tile = analysis.best_understudied(ctx, sites)
    if tile is None:
        out += ["**Best under-studied site.** None among the distinct candidates: every good match lies near a "
                "catalogued analog or in a region the challenge names as well studied.", ""]
    else:
        rank = int(np.flatnonzero(sites == tile)[0]) + 1
        pct = 100 * float(math_engine.earth_percentile(rk.log_score[[tile]], ctx.pair_gaps, sigma, a.weights)[0])
        closer = math_engine.closer_than_share(a.w1[tile], ctx.pair_gaps)
        profile = profile_table(ds.quantiles[tile], ds.target_quantiles[t], rk.log_sim[tile], ds.levels, closer)
        out += [f"**Best under-studied site: {named(tile)}** "
                f"(rank {rank} among distinct sites). It is closer to the target than {pct:.0f} % of random pairs of "
                f"Earth places and lies {ctx.known_dist_km[tile]:.0f} km from the nearest catalogued analog "
                f"({known_names[int(ctx.known_idx[tile])]}). "
                f"[Satellite imagery]({worldview_url(ds.tile_lat[tile], ds.tile_lon[tile])}).", ""]
        out += [f"- {s}" for s in limits_statements(rk.log_sim[tile], profile, closer)]
        tz = analysis.tile_zoom(ctx, tile)
        zoom = None if tz is None else analysis.zoom_similarity(ctx, t, tz.quantiles, a.weights, rk.log_sim[tile])
        if zoom is not None and zoom.order.size:
            best = int(zoom.order[0])
            south, north, west, east = tz.bounds()[best]
            where = coords_text((south + north) / 2, (west + east) / 2, 2)
            figures = (f"topographic similarity {zoom.similarity[best]:.2f}, against {zoom.tile_similarity:.2f} for "
                       "the whole site")
            out.append(f"- Zoom: inside this site, the area of about 11 km centred on {where} has the terrain "
                       f"closest to the target ({figures})." if zoom.similarity[best] >= zoom.tile_similarity else
                       f"- Zoom: no area of about 11 km matches the target's terrain as well as the site taken as a "
                       f"whole; the closest is centred on {where} ({figures}). The match comes from the site's mix "
                       "of terrains.")
        plan = site_test_plan(rk.log_sim[tile], dim_weights, body, closer, target_ice_note(target["target_id"]))
        for section in PLAN_SECTIONS[:3]:
            if plan[section]:
                out.append(f"- {section}: " + "; ".join(item.rstrip(".") for item in plan[section]) + ".")
        when = lighting_calendar_sentence(analysis.lighting_calendar(ctx, t, tile))
        if when:
            out.append(f"- When to go, for the lighting only: {when[0].lower() + when[1:]}")
        out.append("")

    pool, sims, curve = analysis.complementary_sets(ctx, a.w1, a.weights, k_max=2)
    pair = curve[-1]
    labels = [lower_first(config.VARIABLE_BY_NAME[v].label) for v in config.VARIABLE_NAMES]
    parts = []
    for pos, row in enumerate(pair.indices):
        best_for = [labels[k] for k, site_pos in enumerate(pair.best_site_per_variable)
                    if int(site_pos) == pos and np.isfinite(pair.best_per_variable[k]) and a.weights[k] > 0]
        parts.append(f"{named(int(pool[row]))}, best of the pair for: {', '.join(best_for) or 'nothing'}")
    missing = [lower_first(label) for label in uncovered_variables(pair.uncovered, a.weights)]
    out += ["**Two sites that complete each other.** " + "; and ".join(parts) + f". Together they cover "
            f"{pair.coverage:.2f} of the weighted variables (1 = a perfect match on all of them), against "
            f"{pair.best_single:.2f} for the best single site."
            + (f" The pair does not cover: {', '.join(missing)}." if missing else ""), ""]

    out += ["**Can the ranking be trusted here?** "
            + _credibility_sentence(cred, body, ds.known_analogs, ds.targets, p_holm), ""]
    ranked = credibility_table(cred, ds.known_analogs, ds.quantiles, ds.target_quantiles[t], ds.levels)
    if len(ranked):
        out += ["Each tested analog, with the reason for its rank read from the data (reported as is):", "",
                markdown_table(ranked[["Known analog", "Tested on", "Top % (own properties)", WHY_COLUMN]]), ""]

    feature = config.SUBSURFACE_TARGETS.get(str(target["target_id"]))
    if feature is not None and feature.temperature_k is not None and ds.subsurface is not None:
        caves = subsurface_table(ds.subsurface, feature.temperature_k).head(top)
        out += [f"**Below the surface.** {feature.geometry} {feature.temperature_note} {config.CAVE_TEMPERATURE_RULE} "
                "The lava tubes whose surface climate is closest to that temperature are:", "",
                markdown_table(caves[["Lava-tube field", "Country", "Mean annual air temperature (K)",
                                      SUBSURFACE_GAP]]), "",
                f"Case studies, outside the score. Sources: {feature.source}", ""]
    return out, summary


def written_atlas(ctx, dim_weights: Mapping[str, float], mission_label: str, top: int = 3) -> str:
    """The findings as a written document for the public (challenge text: 'documenting your findings in a
    written format'): one section per target, generated by rules from the dataset (no language model).

    ctx: analysis.Context; dim_weights: dimension weights of the mission; top: sites listed per target.
    """
    ds = ctx.ds
    sections, summary = [], []
    creds, p_holm = analysis.credibility_all(ctx)
    for t in range(len(ds.targets)):
        lines, row = _atlas_target(ctx, t, dim_weights, top, creds[t], float(p_holm[t]))
        sections += lines
        summary.append(row)
    out = ["# AnalogExplorer field atlas", "",
           "Where on Earth to rehearse the Moon and Mars, and what Earth cannot teach.", "",
           f"Mission profile: **{mission_label}**. Data: {ds.meta.get('source', 'unknown')}, {ds.n_tiles:,} land "
           f"tiles of Earth compared with {len(ds.targets)} lunar and Martian targets. This document is generated by "
           "rules from the dataset (no language model writes it): `python -m src.report --atlas`.", ""]
    out += ["## How to read this atlas", "",
            "- An **analog** is a place on Earth that reproduces some conditions of a lunar or Martian site. No "
            "place reproduces them all, so each site comes with what matches and what does not.",
            "- **Closer than X % of Earth pairs**: the site is more similar to the target than X % of random pairs "
            "of places on Earth are to each other. 50 % means 'as alike as two ordinary places on Earth'. For one "
            "variable, pairs identical on it (e.g. two places without any bare ground) are left out, and 50 % "
            "always means a gap of one tolerance, the unit of the score.",
            "- **Relative score**: 1 for the best Earth tile for this target and mission.",
            "- Coordinates are tile centres; a tile is about 110 km wide (narrower in longitude near the poles). "
            "A site is named after the country and the physical region of its tile centre (Natural Earth; a reading "
            "aid, outside the score). "
            "Every share of Earth ('closer than X % of Earth pairs', 'top X %') is a share of land area: a tile "
            "counts for its land surface, so the many narrow tiles near the poles do not weigh more than their area.",
            "- Evaluated here: topography, temperature, water and ice, illumination (with the height of the Sun), "
            "bare ground. Not evaluated: "
            + "; ".join(item[0].lower() + item[1:] for item in config.NOT_EVALUATED) + ".", "",
            "## Summary", "", markdown_table(pd.DataFrame(summary)), ""]
    twins = twin_target_sentences(ds.targets["name"].tolist(), analysis.atlas(ctx, dim_weights).top_overlap)
    out += [*[f"- {s}" for s in twins], ""] if twins else []
    out += ["## Targets", ""] + sections
    out += ["## What Earth can never reproduce", "",
            "These limits hold for every site, whatever its score:", "", markdown_table(structural_limits_table()), "",
            "## How the ranking works, in plain words", ""] + [f"- {line}" for line in ATLAS_METHOD] + [""]
    out += ["## Data and references", ""] + [f"- {k}: {v}" for k, v in ds.meta.get("sources", {}).items()]
    out += [f"- {ds.meta['resolution_note']}"] if ds.meta.get("resolution_note") else []
    out += ["- Known analogs: sourced catalogue of the project (Preston & Dartnell 2014, Foucher et al. 2021, NASA "
            "Artemis and ESA PANGAEA training sites), used for the credibility check and the under-studied flag.",
            "- Mission weights and thresholds are hypotheses to calibrate; the interactive tool shows how stable "
            "each result is when they change.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    """Write the atlas to a file: python -m src.report --atlas docs/atlas.md [--mission "Rover Mobility"]."""
    import argparse
    from pathlib import Path

    from src import data_prep

    parser = argparse.ArgumentParser(description="Write the field atlas (Markdown) from the dataset on disk.")
    parser.add_argument("--atlas", required=True, help="output Markdown file")
    parser.add_argument("--mission", default=ATLAS_DEFAULT_MISSION, choices=list(config.MISSION_PROFILES))
    args = parser.parse_args(argv)
    ctx = analysis.build_context(data_prep.load_default_dataset())
    text = written_atlas(ctx, analysis.mission_dimension_weights(args.mission), args.mission)
    Path(args.atlas).write_text(text, encoding="utf-8", newline="\n")
    print(f"Atlas: {len(text.splitlines())} lines -> {args.atlas}")


if __name__ == "__main__":
    main()
