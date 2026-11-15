"""Plotly figures for the Streamlit app. Each function returns a figure; none touches Streamlit.

plotly is imported lazily-safe: if it is missing, HAS_PLOTLY is False and the app falls back
to tables.
"""
from __future__ import annotations

import numpy as np

from src import config

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:  # pragma: no cover - depends on the environment
    go = None
    HAS_PLOTLY = False

INK = "#1B2430"        # reference / text
BLUE = "#2F6FDE"       # candidate sites
AMBER = "#D98A1F"      # highlighted (selected / non-dominated)
GREY = "#9AA5B1"       # context


def _require_plotly() -> None:
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for charts: pip install plotly")


def radar(labels: list[str], similarities: np.ndarray, name: str = "Candidate site",
          strong: np.ndarray | None = None):
    """Radar of per-variable similarity (1 = identical to the target) with the strong-match threshold.

    strong: (K,) similarity above which each variable is a strong match (math_engine.similarity_at_closer_share):
    the status is a share of Earth pairs, so the threshold differs per variable. Without it, the fallback
    threshold on the similarity itself (config.STATUS_STRONG) is drawn as a circle.
    """
    _require_plotly()
    theta = list(labels) + [labels[0]]
    values = list(map(float, similarities)) + [float(similarities[0])]
    ring = ([config.STATUS_STRONG] * len(theta) if strong is None
            else list(map(float, strong)) + [float(strong[0])])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ring, theta=theta, mode="lines", name="Strong-match threshold",
                                  line=dict(color=GREY, dash="dot", width=1)))
    fig.add_trace(go.Scatterpolar(r=values, theta=theta, fill="toself", name=name,
                                  line=dict(color=BLUE, width=2), fillcolor="rgba(47,111,222,0.25)"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0])),
                      showlegend=True, height=380, margin=dict(l=40, r=40, t=30, b=30),
                      legend=dict(orientation="h", y=-0.1))
    return fig


def similarity_novelty_scatter(score: np.ndarray, novelty_: np.ndarray, pareto: np.ndarray,
                               selected: int | None = None):
    """Candidate sites in the similarity/novelty plane; non-dominated sites highlighted."""
    _require_plotly()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=score[~pareto], y=novelty_[~pareto], mode="markers", name="Candidate sites",
                             marker=dict(size=7, color=GREY, opacity=0.7),
                             hovertemplate="score %{x:.3f}<br>novelty %{y:.2f}<extra></extra>"))
    order = np.argsort(score[pareto])
    fig.add_trace(go.Scatter(x=score[pareto][order], y=novelty_[pareto][order], mode="lines+markers",
                             name="Non-dominated", marker=dict(size=9, color=AMBER),
                             line=dict(color=AMBER, width=1),
                             hovertemplate="score %{x:.3f}<br>novelty %{y:.2f}<extra></extra>"))
    if selected is not None:
        fig.add_trace(go.Scatter(x=[score[selected]], y=[novelty_[selected]], mode="markers", name="Selected",
                                 marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=INK, width=2))))
    fig.update_layout(xaxis_title="Similarity score (geometric mean)", yaxis_title="Novelty",
                      height=380, margin=dict(l=50, r=20, t=30, b=50), legend=dict(orientation="h", y=-0.2))
    return fig


def coverage_heatmap(sim_matrix: np.ndarray, site_labels: list[str], var_labels: list[str]):
    """Similarity of each selected site (rows) on each variable (columns)."""
    _require_plotly()
    fig = go.Figure(go.Heatmap(z=sim_matrix, x=var_labels, y=site_labels, zmin=0, zmax=1,
                               colorscale="Blues", texttemplate="%{z:.2f}",
                               colorbar=dict(title="Similarity")))
    fig.update_layout(height=120 + 60 * len(site_labels), margin=dict(l=20, r=20, t=20, b=20),
                      yaxis=dict(autorange="reversed"))
    return fig


def coverage_curve(ks: list[int], coverage: list[float], ceiling: float):
    """Coverage as a function of the number of sites, against the best-per-variable ceiling."""
    _require_plotly()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=coverage, mode="lines+markers", name="Best set of k sites",
                             line=dict(color=BLUE, width=2), marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=[min(ks), max(ks)], y=[ceiling, ceiling], mode="lines",
                             name="Ceiling (one site per variable)", line=dict(color=GREY, dash="dot")))
    fig.update_layout(xaxis=dict(title="Number of sites", tickvals=ks), yaxis=dict(title="Coverage", range=[0, 1.02]),
                      height=300, margin=dict(l=50, r=20, t=20, b=50), legend=dict(orientation="h", y=-0.3))
    return fig


def distribution_compare(site_q: np.ndarray, target_q: np.ndarray, levels: np.ndarray,
                         labels: list[str], units: list[str], n_cols: int = 4):
    """Quantile curves of the site against the target, one panel per variable.

    The shaded area between the two curves is proportional to the Wasserstein distance
    used by the engine, so the chart shows *where* the distributions differ.
    """
    _require_plotly()
    from plotly.subplots import make_subplots

    k = len(labels)
    n_rows = int(np.ceil(k / n_cols))
    # two lines per title: long labels do not overlap
    titles = [f"{label}<br>({u})" for label, u in zip(labels, units, strict=True)]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        horizontal_spacing=0.07, vertical_spacing=0.22)
    pct = 100 * np.asarray(levels)
    for i in range(k):
        r, c = i // n_cols + 1, i % n_cols + 1
        first = i == 0
        fig.add_trace(go.Scatter(x=pct, y=target_q[i], mode="lines", name="Target", legendgroup="t",
                                 showlegend=first, line=dict(color=INK, dash="dash", width=2)), row=r, col=c)
        fig.add_trace(go.Scatter(x=pct, y=site_q[i], mode="lines", name="Site", legendgroup="s",
                                 showlegend=first, fill="tonexty", fillcolor="rgba(217,138,31,0.25)",
                                 line=dict(color=BLUE, width=2)), row=r, col=c)
    fig.update_xaxes(title_text="Quantile (%)", title_font=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    fig.update_annotations(font_size=10)
    fig.update_layout(height=280 * n_rows, margin=dict(l=30, r=10, t=50, b=40),
                      legend=dict(orientation="h", y=-0.12))
    return fig


def ensemble_radar(sim_matrix: np.ndarray, site_labels: list[str], var_labels: list[str]):
    """One radar trace per site of the ensemble, plus the coverage envelope (best site per variable)."""
    _require_plotly()
    theta = list(var_labels) + [var_labels[0]]
    palette = [BLUE, AMBER, "#3BA272"]
    fig = go.Figure()
    envelope = np.asarray(sim_matrix).max(axis=0)
    fig.add_trace(go.Scatterpolar(r=list(envelope) + [envelope[0]], theta=theta, fill="toself",
                                  name="Covered by the set", line=dict(color=GREY, width=1),
                                  fillcolor="rgba(154,165,177,0.25)"))
    for i, label in enumerate(site_labels):
        row = np.asarray(sim_matrix[i], dtype=float)
        fig.add_trace(go.Scatterpolar(r=list(row) + [row[0]], theta=theta, mode="lines+markers", name=label,
                                      line=dict(color=palette[i % len(palette)], width=2)))
    fig.add_trace(go.Scatterpolar(r=[config.COVERED_THRESHOLD] * len(theta), theta=theta, mode="lines",
                                  name="Covered threshold", line=dict(color=INK, dash="dot", width=1)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0])),
                      height=430, margin=dict(l=40, r=40, t=30, b=30), legend=dict(orientation="h", y=-0.1))
    return fig


def hillshade(dem: np.ndarray, dy_m: float, dx_m: float | np.ndarray, azimuth_deg: float = 315.0,
              altitude_deg: float = 35.0) -> np.ndarray:
    """Shaded relief in [0, 1] of a north-up DEM (NaN where it has no data).

    Cosine of the angle between the surface normal and the light direction (azimuth clockwise
    from north, altitude above the horizon). dx_m: one value, or one per row.
    """
    z = np.asarray(dem, dtype=np.float64)
    dz_north = -np.gradient(z, axis=0) / float(dy_m)                      # row index grows southward
    dz_east = np.gradient(z, axis=1) / np.asarray(dx_m, dtype=np.float64).reshape(-1, 1)
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    light = np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)
    shade = (-dz_east * light[0] - dz_north * light[1] + light[2]) / np.sqrt(1.0 + dz_east ** 2 + dz_north ** 2)
    return np.where(np.isfinite(z), np.clip(shade, 0.0, 1.0), np.nan)   # a central difference skips its own pixel


def zoom_figure(site_dem: np.ndarray, site_px_m: tuple, target_dem: np.ndarray | None, sub_bounds: np.ndarray,
                sub_similarity: np.ndarray, grid: tuple[int, int], best: int | None = None,
                target_px_m: float = 463.0, titles: tuple[str, str, str] = ("Earth site", "Sub-areas", "Target")):
    """Fine-scale zoom of a site: its shaded relief, the similarity of its sub-areas, the target's relief.

    The three panels share one ground scale (km). site_px_m: (dy, dx per row) in metres;
    sub_bounds: (S, 4) south, north, west, east of each sub-area as fractions of the tile;
    sub_similarity: (S,) in [0, 1] (NaN = not enough land); best: sub-area to outline.
    """
    _require_plotly()
    from plotly.subplots import make_subplots

    dy, dx = site_px_m
    rows, cols = site_dem.shape
    height, width = rows * dy / 1000.0, cols * float(np.mean(dx)) / 1000.0
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(titles), horizontal_spacing=0.06)
    relief = dict(colorscale="Greys_r", zmin=0.0, zmax=1.0, showscale=False, hoverinfo="skip")
    fig.add_trace(go.Heatmap(z=hillshade(site_dem, dy, dx)[::-1], x0=width / cols / 2, dx=width / cols,
                             y0=height / rows / 2, dy=height / rows, **relief), row=1, col=1)
    s, n, w, e = (np.asarray(sub_bounds)[:, i] for i in range(4))
    block_h, block_w = height / grid[0], width / grid[1]                  # sub-areas are near-uniform blocks
    fig.add_trace(go.Heatmap(z=np.asarray(sub_similarity, dtype=float).reshape(grid)[::-1], x0=block_w / 2,
                             dx=block_w, y0=block_h / 2, dy=block_h, colorscale="Blues", zmin=0.0, zmax=1.0,
                             colorbar=dict(title="Similarity", len=0.8),
                             hovertemplate="similarity %{z:.2f}<extra></extra>"), row=1, col=2)
    if best is not None:
        for col in (1, 2):
            fig.add_shape(type="rect", x0=w[best] * width, x1=e[best] * width, y0=s[best] * height,
                          y1=n[best] * height, line=dict(color=AMBER, width=3), row=1, col=col)
    extent = max(height, width)
    if target_dem is not None:
        t_rows, t_cols = target_dem.shape
        px_km = target_px_m / 1000.0
        fig.add_trace(go.Heatmap(z=hillshade(target_dem, target_px_m, target_px_m)[::-1], x0=px_km / 2, dx=px_km,
                                 y0=px_km / 2, dy=px_km, **relief), row=1, col=3)
        extent = max(extent, t_rows * px_km, t_cols * px_km)
    for col in (1, 2, 3):
        fig.update_xaxes(range=[0, extent], title_text="km", constrain="domain", row=1, col=col)
        fig.update_yaxes(range=[0, extent], scaleanchor="x" if col == 1 else f"x{col}", scaleratio=1,
                         constrain="domain", row=1, col=col)
    fig.update_layout(height=420, margin=dict(l=30, r=10, t=50, b=40), plot_bgcolor="#DCE6F0")
    return fig


TARGET_COLORS = ("#0072B2", "#56B4E9", "#009E73", "#D55E00", "#E69F00", "#CC79A7", "#000000",
                 "#F0E442")                                              # Okabe-Ito, colour-blind safe


def atlas_map(lat: np.ndarray, lon: np.ndarray, best: np.ndarray, percentile: np.ndarray,
              target_names: list[str], selected: tuple[float, float] | None = None,
              places: np.ndarray | None = None, size: int = 4):
    """World map of the Earth tiles coloured by the target they resemble most.

    lat, lon, best, percentile: one entry per tile to draw (already filtered); best indexes
    target_names; percentile in [0, 1] is shown on hover. selected: optional (lat, lon) to ring.
    places: optional name of each tile, shown on hover; size: marker size.
    """
    _require_plotly()
    fig = go.Figure()
    for t, name in enumerate(target_names):
        mask = np.asarray(best) == t
        where = None if places is None else np.asarray(places, dtype=object)[mask]
        fig.add_trace(go.Scattergeo(
            lat=np.asarray(lat)[mask], lon=np.asarray(lon)[mask], mode="markers", name=f"{name} ({int(mask.sum())})",
            customdata=100 * np.asarray(percentile)[mask], text=where,
            marker=dict(size=size, color=TARGET_COLORS[t % len(TARGET_COLORS)], opacity=0.8),
            hovertemplate=name + ("<br>%{text}" if where is not None else "")
            + "<br>lat %{lat:.1f}, lon %{lon:.1f}<br>closer than %{customdata:.0f} % of Earth pairs<extra></extra>"))
    if selected is not None:
        fig.add_trace(go.Scattergeo(lat=[selected[0]], lon=[selected[1]], mode="markers", name="Selected place",
                                    marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=INK, width=2))))
    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#E8ECEF", showocean=True,
                    oceancolor="#F7FAFC", showcountries=False, lataxis_range=[-90, 90], lonaxis_range=[-180, 180])
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.05))
    return fig


def sites_map(lat: np.ndarray, lon: np.ndarray, score: np.ndarray, highlight: list[int] | None = None,
              known: tuple[np.ndarray, np.ndarray, list[str]] | None = None, places: np.ndarray | None = None,
              numbered: int = 0, score_title: str = "Score"):
    """World map of the candidate sites (colour = score); highlighted rows are ringed.

    known: optional (lat, lon, names) of catalogued analogs, drawn as diamonds. places: optional name of
    each site, shown on hover. numbered: write the rank (1, 2, ...) next to that many first sites.
    """
    _require_plotly()
    fig = go.Figure(go.Scattergeo(
        lat=lat, lon=lon, mode="markers", name="Candidate sites", text=places,
        marker=dict(size=8, color=score, colorscale="Blues", cmin=0, cmax=float(max(score.max(), 1e-9)),
                    showscale=True, colorbar=dict(title=score_title)),
        hovertemplate=("%{text}<br>" if places is not None else "")
        + "lat %{lat:.2f}, lon %{lon:.2f}<br>score %{marker.color:.2f}<extra></extra>"))
    if numbered:
        n = min(int(numbered), len(lat))
        fig.add_trace(go.Scattergeo(lat=np.asarray(lat)[:n], lon=np.asarray(lon)[:n], mode="text",
                                    text=[str(i + 1) for i in range(n)], textposition="top center",
                                    textfont=dict(size=13, color=INK), hoverinfo="skip", showlegend=False))
    if highlight:
        h = np.asarray(highlight, dtype=int)
        fig.add_trace(go.Scattergeo(lat=lat[h], lon=lon[h], mode="markers", name="Selected",
                                    marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=AMBER, width=3))))
    if known is not None:
        k_lat, k_lon, k_names = known
        fig.add_trace(go.Scattergeo(lat=k_lat, lon=k_lon, mode="markers", name="Known analogs", text=k_names,
                                    marker=dict(size=9, symbol="diamond", color=INK),
                                    hovertemplate="%{text}<extra></extra>"))
    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#E8ECEF",
                    showocean=True, oceancolor="#F7FAFC", showcountries=False)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.05))
    return fig
