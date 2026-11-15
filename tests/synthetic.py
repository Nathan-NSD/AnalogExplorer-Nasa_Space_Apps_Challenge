"""Synthetic dataset in the final data format, for the tests only.

Small, fast and deterministic, it lets the tests exercise the engine, the report and every page of the app
without network nor data/. Everything here is INVENTED: target profiles are not real measurements. The app and
the atlas never load it (they stop when the real dataset is missing).
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import beta, gamma, norm

from src import config, data_prep, math_engine

# Variable parameters are given in config.VARIABLE_NAMES order via these dictionaries.
SHAPES = {
    # name: (distribution, shape parameter, noise kind, noise scale)
    "slope_deg": ("gamma", 2.0, "mult", 0.12),      # theta = gamma scale (deg)
    "roughness_m": ("gamma", 2.0, "mult", 0.12),    # theta = gamma scale (m)
    "diurnal_amp_k": ("norm", 3.0, "add", 2.0),     # theta = mean (K), shape = std (K)
    "temp_min_k": ("norm", 4.0, "add", 3.0),        # theta = mean (K), shape = std (K)
    "aridity": ("beta", 12.0, "add", 0.04),         # theta = mean in (0, 1), shape = concentration
    "ice_proxy": ("beta", 12.0, "add", 0.04),
    "insolation_wm2": ("norm", 25.0, "add", 15.0),  # theta = mean (W/m2), shape = std (W/m2)
    "sunlit_frac": ("beta", 20.0, "add", 0.03),
    "sun_elevation_deg": ("norm", 6.0, "add", 3.0),  # theta = mean (deg), shape = std (deg)
    "bare_frac": ("beta", 12.0, "add", 0.04),
}

ARCHETYPES = {
    "hot_desert": {"slope_deg": 1.2, "roughness_m": 0.25, "diurnal_amp_k": 32, "temp_min_k": 280,
                   "aridity": 0.95, "ice_proxy": 0.02, "insolation_wm2": 270, "sunlit_frac": 0.42,
                   "sun_elevation_deg": 42, "bare_frac": 0.90},
    "polar_desert": {"slope_deg": 1.8, "roughness_m": 0.35, "diurnal_amp_k": 18, "temp_min_k": 235,
                     "aridity": 0.80, "ice_proxy": 0.65, "insolation_wm2": 95, "sunlit_frac": 0.30,
                     "sun_elevation_deg": 14, "bare_frac": 0.55},
    "volcanic": {"slope_deg": 5.5, "roughness_m": 1.20, "diurnal_amp_k": 24, "temp_min_k": 270,
                 "aridity": 0.65, "ice_proxy": 0.08, "insolation_wm2": 210, "sunlit_frac": 0.32,
                 "sun_elevation_deg": 36, "bare_frac": 0.70},
    "temperate": {"slope_deg": 1.0, "roughness_m": 0.20, "diurnal_amp_k": 12, "temp_min_k": 275,
                  "aridity": 0.15, "ice_proxy": 0.01, "insolation_wm2": 150, "sunlit_frac": 0.20,
                  "sun_elevation_deg": 30, "bare_frac": 0.05},
}

# Invented profiles. Coordinates are approximate (display only). No single archetype
# matches a target on every dimension, so complementary sets are meaningful in the tests.
# Illumination orders of magnitude: polar lunar ridges are lit most of the time but at
# grazing incidence (low flux on a horizontal surface); the lunar equator gets the full
# solar constant with no atmosphere, which no place on Earth reproduces.
TARGETS = (
    dict(target_id="moon_shackleton", name="Shackleton-de Gerlache Ridge (South Pole)", body="Moon",
         lat=-89.9, lon=0.0,
         theta=dict(slope_deg=6.0, roughness_m=1.3, diurnal_amp_k=22, temp_min_k=228, aridity=0.98, ice_proxy=0.70,
                    insolation_wm2=40, sunlit_frac=0.85, sun_elevation_deg=12, bare_frac=0.98)),
    dict(target_id="moon_malapert", name="Malapert Massif", body="Moon", lat=-86.0, lon=0.1,
         theta=dict(slope_deg=5.0, roughness_m=1.0, diurnal_amp_k=26, temp_min_k=236, aridity=0.98, ice_proxy=0.45,
                    insolation_wm2=55, sunlit_frac=0.75, sun_elevation_deg=14, bare_frac=0.98)),
    dict(target_id="moon_mare_tranq_pit", name="Mare Tranquillitatis (Pit Crater)", body="Moon",
         lat=8.3, lon=33.2,
         theta=dict(slope_deg=1.5, roughness_m=0.5, diurnal_amp_k=34, temp_min_k=272, aridity=0.98, ice_proxy=0.03,
                    insolation_wm2=430, sunlit_frac=0.50, sun_elevation_deg=42, bare_frac=0.98)),
    dict(target_id="mars_jezero", name="Jezero Crater", body="Mars", lat=18.4, lon=77.6,
         theta=dict(slope_deg=2.5, roughness_m=0.6, diurnal_amp_k=30, temp_min_k=240, aridity=0.95, ice_proxy=0.10,
                    insolation_wm2=145, sunlit_frac=0.45, sun_elevation_deg=40, bare_frac=0.98)),
    dict(target_id="mars_gale", name="Gale Crater", body="Mars", lat=-5.4, lon=137.8,
         theta=dict(slope_deg=4.0, roughness_m=0.9, diurnal_amp_k=28, temp_min_k=245, aridity=0.93, ice_proxy=0.12,
                    insolation_wm2=160, sunlit_frac=0.46, sun_elevation_deg=42, bare_frac=0.98)),
    dict(target_id="mars_arcadia", name="Arcadia Planitia (Ice-rich)", body="Mars", lat=47.0, lon=-175.0,
         theta=dict(slope_deg=1.0, roughness_m=0.25, diurnal_amp_k=18, temp_min_k=232, aridity=0.85, ice_proxy=0.75,
                    insolation_wm2=100, sunlit_frac=0.40, sun_elevation_deg=28, bare_frac=0.98)),
    dict(target_id="mars_oxia", name="Oxia Planum (Rosalind Franklin landing site)", body="Mars", lat=18.3,
         lon=-24.6,
         theta=dict(slope_deg=1.2, roughness_m=0.3, diurnal_amp_k=31, temp_min_k=242, aridity=0.95, ice_proxy=0.08,
                    insolation_wm2=140, sunlit_frac=0.45, sun_elevation_deg=40, bare_frac=0.98)),
)

# Real, well-known analog sites with approximate coordinates (the app uses config.KNOWN_ANALOGS).
# relevant_to: bodies this site is commonly used as an analog for (";"-separated).
KNOWN_ANALOGS = (
    ("Atacama Desert", -24.5, -69.2, "Mars"), ("Haughton Crater (Devon Island)", 75.4, -89.7, "Moon;Mars"),
    ("Death Valley", 36.5, -117.0, "Mars"), ("McMurdo Dry Valleys", -77.5, 162.5, "Mars"),
    ("Rio Tinto", 37.7, -6.5, "Mars"), ("Mauna Kea (Hawaii)", 19.8, -155.5, "Moon;Mars"),
    ("Iceland highlands", 64.9, -18.6, "Moon;Mars"), ("Svalbard", 78.2, 15.6, "Mars"),
)


def _quantiles_from_theta(theta: np.ndarray, variables: tuple[str, ...], levels: np.ndarray) -> np.ndarray:
    """Quantile vectors (n, K, L) from per-tile parameters theta (n, K), using SHAPES."""
    n = theta.shape[0]
    out = np.empty((n, len(variables), levels.size), dtype=np.float32)
    for j, name in enumerate(variables):
        kind, shape, _, _ = SHAPES[name]
        t = theta[:, j:j + 1]
        if kind == "gamma":
            out[:, j, :] = t * gamma.ppf(levels, a=shape)[None, :]
        elif kind == "norm":
            out[:, j, :] = t + shape * norm.ppf(levels)[None, :]
        else:  # beta with mean t and concentration `shape`
            out[:, j, :] = beta.ppf(levels[None, :], t * shape, (1.0 - t) * shape)
    return out


def build_dataset(n_clusters: int = 160, tiles_per_cluster: int = 45, seed: int = 42,
                  with_zoom: bool = False) -> data_prep.Dataset:
    """Synthetic world in the final data format. Deterministic for a given seed.

    Tiles come in spatial clusters (like real terrain, neighbouring tiles look alike), each
    cluster being a Dirichlet mixture of four invented archetypes. Not a scientific result.
    with_zoom: also invent the optional layers (relief of the best sites, cave temperatures), so that
    their views can be exercised.
    """
    rng = np.random.default_rng(seed)
    variables = config.VARIABLE_NAMES
    levels = config.QUANTILE_LEVELS.astype(np.float64)
    arch = np.array([[a[v] for v in variables] for a in ARCHETYPES.values()], dtype=np.float64)

    s_lo, s_hi = np.sin(np.radians([-70.0, 80.0]))
    c_lat = np.degrees(np.arcsin(rng.uniform(s_lo, s_hi, n_clusters)))
    c_lon = rng.uniform(-180.0, 180.0, n_clusters)
    mix = rng.dirichlet(np.full(len(arch), 0.35), n_clusters)
    cluster = np.repeat(np.arange(n_clusters), tiles_per_cluster)
    n = cluster.size

    lat = np.clip(c_lat[cluster] + rng.normal(0.0, 1.5, n), -89.9, 89.9)
    lon_scatter = rng.normal(0.0, 1.5, n) / np.maximum(np.cos(np.radians(lat)), 0.2)
    lon = (c_lon[cluster] + lon_scatter + 180.0) % 360.0 - 180.0

    theta = (mix @ arch)[cluster]
    for j, name in enumerate(variables):
        kind, _, noise_kind, noise = SHAPES[name]
        theta[:, j] = theta[:, j] * rng.lognormal(0.0, noise, n) if noise_kind == "mult" \
            else theta[:, j] + rng.normal(0.0, noise, n)
        if kind == "beta":
            theta[:, j] = np.clip(theta[:, j], 0.01, 0.99)
    quantiles = _quantiles_from_theta(theta, variables, levels)

    target_theta = np.array([[t["theta"][v] for v in variables] for t in TARGETS], dtype=np.float64)
    for j, name in enumerate(variables):
        if SHAPES[name][0] == "beta":
            target_theta[:, j] = np.clip(target_theta[:, j], 0.01, 0.99)
    target_q = _quantiles_from_theta(target_theta, variables, levels)

    targets = pd.DataFrame([{k: t[k] for k in ("target_id", "name", "body", "lat", "lon")} for t in TARGETS])
    targets["note"] = "Invented profile (test data)."
    known = pd.DataFrame(KNOWN_ANALOGS, columns=["name", "lat", "lon", "relevant_to"])
    known["source"] = "Approximate coordinates (test catalogue)"

    ds = data_prep.Dataset(
        variables=variables, levels=levels.astype(np.float32),
        tile_id=np.arange(n, dtype=np.int64), tile_lat=lat.astype(np.float32), tile_lon=lon.astype(np.float32),
        valid_frac=(valid := rng.uniform(0.9, 1.0, n).astype(np.float32)), quantiles=quantiles,
        sigma_default=math_engine.default_tolerances(quantiles, weights=data_prep.tile_area_weights(lat, valid)),
        targets=targets, target_quantiles=target_q, known_analogs=known,
        meta={"source": "synthetic", "version": 2, "seed": seed, "n_tiles": int(n),
              "variables": list(variables), "generated_at": datetime.now(UTC).isoformat(),
              "notes": "Synthetic tiles and invented target profiles. Not a scientific result."},
    )
    ds.validate()
    if with_zoom:
        ds.zoom, ds.target_dems = _invented_zoom(ds, seed)
        ds.subsurface = _invented_subsurface()
    return ds


def _invented_zoom(ds: data_prep.Dataset, seed: int = 42, n_sites: int = 3) -> tuple[dict, dict]:
    """INVENTED relief (smoothed noise) for the best synthetic sites and the targets, in the zoom format."""
    rng = np.random.default_rng(seed)

    def relief(shape: tuple[int, int]) -> np.ndarray:
        return (2500.0 * ndimage.gaussian_filter(rng.normal(size=shape), 4.0)).astype(np.float32)

    zoom = {}
    for lat0, lon0 in data_prep.default_zoom_tiles(ds, n_sites):
        dem = relief((data_prep.TOPO_PX_PER_DEG, data_prep.tile_columns(lat0)))
        tile = data_prep.TileZoom(lat0, lon0, dem, np.empty(0), np.empty(0), (0, 0))
        tile.quantiles, tile.land, tile.grid = data_prep.subarea_topography(dem, tile.pixel_size_m())
        zoom[(lat0, lon0)] = tile
    n = int(round(2 * data_prep.TARGET_HALF_KM * 1000 / data_prep.TOPO_PIXEL_M))
    return zoom, {tid: relief((n, n)) for tid in ds.targets["target_id"]}


def _invented_subsurface() -> pd.DataFrame:
    """The same catalogue with an INVENTED temperature (a function of latitude), for the synthetic dataset."""
    rows = [(name, country, lat, lon, round(300.0 - 0.55 * abs(lat), 2), "Invented temperature (test data)")
            for name, country, lat, lon in config.LAVA_TUBE_SITES]
    return pd.DataFrame(rows, columns=list(data_prep.SUBSURFACE_COLUMNS))
