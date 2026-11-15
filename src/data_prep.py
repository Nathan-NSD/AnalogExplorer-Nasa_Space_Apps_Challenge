"""Offline data preparation.

Builds the real dataset the engine and the app read, following the data contract of
docs/data.md, section 4 (tiles.npz, targets.csv, known_analogs.csv, meta.json and the
optional zoom.npz and subsurface.csv): Earth tiles (Copernicus DEM, NASA POWER) and planetary
targets (PDS, PGDA, Mars Climate Database), with pure numpy raster blocks (slope, roughness,
quantiles). The tests build a small synthetic dataset in the same format (tests/synthetic.py).

This module never imports Streamlit, so it runs as an offline script:
    python -m src.data_prep --real | --targets | --layers
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from src import config, math_engine

ROOT = Path(__file__).resolve().parents[1]
DATASET_ENV_VAR = "ANALOG_DATASET_DIR"
DEFAULT_DATASET_DIR = ROOT / "data" / "processed" / "real_v1"   # the versioned dataset of the app


# ===================================================================== dataset
@dataclass
class Dataset:
    """Everything the engine needs, already harmonised (see the data contract)."""

    variables: tuple[str, ...]
    levels: np.ndarray                 # (L,)
    tile_id: np.ndarray                # (n,)
    tile_lat: np.ndarray               # (n,)
    tile_lon: np.ndarray               # (n,)
    valid_frac: np.ndarray             # (n,)
    quantiles: np.ndarray              # (n, K, L) float32
    sigma_default: np.ndarray          # (K,)
    targets: pd.DataFrame              # target_id, name, body, lat, lon, note
    target_quantiles: np.ndarray       # (T, K, L)
    known_analogs: pd.DataFrame        # name, lat, lon, source
    meta: dict = field(default_factory=dict)
    zoom: dict = field(default_factory=dict)         # {(lat0, lon0): TileZoom}, optional
    target_dems: dict = field(default_factory=dict)  # {target_id: elevation window (m)}, optional, for display
    subsurface: pd.DataFrame | None = None           # lava-tube case studies, optional
    places: pd.DataFrame | None = None               # country and region of each tile, optional, display only

    @property
    def n_tiles(self) -> int:
        return int(self.tile_id.shape[0])

    def target_index(self, target_id: str) -> int:
        hits = np.flatnonzero(self.targets["target_id"].to_numpy() == target_id)
        if hits.size == 0:
            raise KeyError(f"unknown target {target_id!r}")
        return int(hits[0])

    def validate(self) -> None:
        """Fail early, with a clear message, when a dataset breaks the contract."""
        n, k, n_lev = self.quantiles.shape
        if tuple(self.variables) != config.VARIABLE_NAMES:
            raise ValueError(f"dataset variables {self.variables} differ from config {config.VARIABLE_NAMES}")
        for name, arr in (("tile_id", self.tile_id), ("tile_lat", self.tile_lat),
                          ("tile_lon", self.tile_lon), ("valid_frac", self.valid_frac)):
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if self.levels.shape != (n_lev,) or self.sigma_default.shape != (k,):
            raise ValueError("levels / sigma_default shapes do not match quantiles")
        if self.target_quantiles.shape != (len(self.targets), k, n_lev):
            raise ValueError("target_quantiles must have shape (n_targets, K, L)")
        if not np.isfinite(self.quantiles).all():
            raise ValueError("tile quantiles contain NaN/inf: drop invalid tiles before saving")
        finite_rows = np.isfinite(self.target_quantiles).all(axis=2)
        if not (finite_rows | np.isnan(self.target_quantiles).all(axis=2)).all():
            raise ValueError("a target variable must be fully measured or fully NaN (not evaluated)")
        if not finite_rows.any(axis=1).all():
            raise ValueError("every target needs at least one measured variable")
        for q in (self.quantiles, self.target_quantiles[finite_rows]):
            if np.any(np.diff(q, axis=-1) < -1e-5):
                raise ValueError("quantiles must be non-decreasing along the level axis")


# ============================================================== raster blocks
def slope_degrees(dem: np.ndarray, pixel_size_m: float | tuple) -> np.ndarray:
    """Slope in degrees from a DEM (central differences). NaN propagate.

    pixel_size_m: one size for square pixels, or (dy, dx) in metres where dx may be an
    array with one value per row (geographic grids: see `geographic_pixel_size`).
    The slope depends on the baseline: compute it at the *same* pixel size on every
    world.
    """
    z = np.asarray(dem, dtype=np.float32)
    dy, dx = (pixel_size_m, pixel_size_m) if np.isscalar(pixel_size_m) else pixel_size_m
    gy = np.gradient(z, axis=0) / np.float32(dy)
    gx = np.gradient(z, axis=1) / np.asarray(dx, dtype=np.float32).reshape(-1, 1)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def geographic_pixel_size(lat_deg: np.ndarray, dlat_deg: float, dlon_deg: float,
                          radius_km: float) -> tuple[float, np.ndarray]:
    """(dy, dx per row) in metres of a latitude/longitude grid on a sphere of `radius_km`.

    lat_deg: latitude of each row centre. dx shrinks with cos(latitude).
    """
    r = radius_km * 1000.0
    dy = r * np.radians(dlat_deg)
    dx = r * np.radians(dlon_deg) * np.cos(np.radians(np.asarray(lat_deg, dtype=np.float64)))
    return float(dy), np.maximum(dx, 1e-3)


def local_roughness(dem: np.ndarray, window_px: int, min_valid_frac: float = 0.5) -> np.ndarray:
    """Local standard deviation of elevation in a square window (NaN-aware, O(N))."""
    z = np.asarray(dem, dtype=np.float64)
    valid = np.isfinite(z)
    if not valid.any():
        return np.full(z.shape, np.nan, dtype=np.float32)
    x = np.where(valid, z - z[valid].mean(), 0.0)      # centred: avoids catastrophic cancellation
    frac = ndimage.uniform_filter(valid.astype(np.float64), size=window_px, mode="nearest")
    safe = np.maximum(frac, 1e-12)
    mean1 = ndimage.uniform_filter(x, size=window_px, mode="nearest") / safe
    mean2 = ndimage.uniform_filter(x * x, size=window_px, mode="nearest") / safe
    out = np.sqrt(np.maximum(mean2 - mean1 * mean1, 0.0))
    out[frac < min_valid_frac] = np.nan
    return out.astype(np.float32)


# ========================================================== real data: Earth tiles
BODY_RADIUS_KM = {"Earth": 6371.0088, "Moon": 1737.4, "Mars": 3389.5}
TILE_DEG = 1                    # Earth tile = one 1 x 1 deg cell (Copernicus DEM and POWER grids)
TOPO_PX_PER_DEG = 240           # 1/240 deg ~ 463 m on Earth: common topographic scale
TOPO_PIXEL_M = 463.0            # same scale for the Moon and Mars windows
ROUGHNESS_WINDOW_PX = 5         # ~2.3 km
MIN_LAND_FRAC = 0.25            # a tile is kept if at least this share of its pixels is land
TOPO_CACHE_DIR = ROOT / "data" / "processed" / "cache" / "topography_wbm"   # water-masked version
TOPO_VARIABLES = ("slope_deg", "roughness_m")
COVER_VARIABLES = ("bare_frac",)
ASTRO_VARIABLES = ("sun_elevation_deg",)      # astronomy only: computed from the latitude, no data set
CLIMATE_VARIABLES = tuple(v for v in config.VARIABLE_NAMES
                          if v not in TOPO_VARIABLES + COVER_VARIABLES + ASTRO_VARIABLES)
# A land tile where water is never liquid (every month below 0 C, e.g. ice sheets, Dry Valleys)
# has no liquid water at all: aridity 1, the same physical value as the Moon and Mars.
CLIMATE_FALLBACK = {"aridity": 1.0}


def sample_quantiles(values: np.ndarray, levels: np.ndarray | None = None) -> np.ndarray:
    """Quantiles (L,) of the finite values (space and/or time samples); NaN when there is none."""
    levels = config.QUANTILE_LEVELS if levels is None else np.asarray(levels)
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return np.quantile(v, levels).astype(np.float32) if v.size else np.full(levels.shape, np.nan, np.float32)


def topography_quantiles(dem_m: np.ndarray, pixel_size_m: float | tuple,
                         levels: np.ndarray | None = None) -> np.ndarray:
    """(2, L) quantiles of slope (deg) and roughness (m) of a DEM at the common topographic scale."""
    slope = slope_degrees(dem_m, pixel_size_m)
    rough = local_roughness(dem_m, ROUGHNESS_WINDOW_PX)
    return np.stack([sample_quantiles(slope, levels), sample_quantiles(rough, levels)])


def tile_columns(lat0: int) -> int:
    """Columns of a 1 x 1 deg Earth tile at the common ~463 m pixel: 240 * cos(latitude), at least 8."""
    return max(8, int(round(TOPO_PX_PER_DEG * np.cos(np.radians(lat0 + 0.5)))))


def earth_tile_dem(name: str, lat0: int) -> tuple[np.ndarray, tuple[float, np.ndarray]]:
    """Elevation (m, NaN = water) of one Copernicus tile and its pixel size (dy, dx per row) in metres.

    The tile is read with about square ~463 m pixels: 240 rows and 240 * cos(lat) columns.
    """
    from src import sources

    nx = tile_columns(lat0)
    dem = sources.read_copernicus_tile(name, (TOPO_PX_PER_DEG, nx))
    rows_lat = lat0 + 1 - (np.arange(TOPO_PX_PER_DEG) + 0.5) / TOPO_PX_PER_DEG     # north-up
    return dem, geographic_pixel_size(rows_lat, 1 / TOPO_PX_PER_DEG, 1 / nx, BODY_RADIUS_KM["Earth"])


def earth_tile_topography(name: str, lat0: int, levels: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Slope/roughness quantiles (2, L) and land fraction of one Copernicus tile."""
    dem, size = earth_tile_dem(name, lat0)
    return topography_quantiles(dem, size, levels), float(np.isfinite(dem).mean())


def cells_in_tile(grid_lat: np.ndarray, grid_lon: np.ndarray, lat0: float, lon0: float,
                  size: float = TILE_DEG) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the grid cells whose centre falls inside a tile (nearest cell if none)."""
    rows = np.flatnonzero((grid_lat >= lat0) & (grid_lat < lat0 + size))
    cols = np.flatnonzero((grid_lon >= lon0) & (grid_lon < lon0 + size))
    if rows.size == 0:
        rows = np.array([np.abs(grid_lat - (lat0 + size / 2)).argmin()])
    if cols.size == 0:
        cols = np.array([np.abs(grid_lon - (lon0 + size / 2)).argmin()])
    return rows, cols


def earth_climate_quantiles(fields: Mapping[str, tuple], lat0: float, lon0: float,
                            levels: np.ndarray | None = None) -> np.ndarray:
    """(len(CLIMATE_VARIABLES), L) quantiles over the grid cells of a tile and the 12 months."""
    out = []
    for name in CLIMATE_VARIABLES:
        field, grid_lat, grid_lon = fields[name]
        rows, cols = cells_in_tile(grid_lat, grid_lon, lat0, lon0)
        q = sample_quantiles(field[:, rows][:, :, cols], levels)
        if np.isnan(q).all() and name in CLIMATE_FALLBACK:
            q = np.full_like(q, CLIMATE_FALLBACK[name])
        out.append(q)
    return np.stack(out)


def _tile_batches(tiles: list, cache_dir: Path, compute, workers: int, batch: int, log, label: str,
                  prefix: str) -> dict:
    """compute(tile) -> (quantiles, land share) for every tile, in parallel, saved batch by batch so a run
    can resume (cache files <prefix>_NNNN.npz with names, q, land)."""
    from concurrent.futures import ThreadPoolExecutor

    cache_dir.mkdir(parents=True, exist_ok=True)
    done: dict[str, tuple[np.ndarray, float]] = {}
    for f in sorted(cache_dir.glob(f"{prefix}_*.npz")):
        z = np.load(f)
        done.update({str(n): (q, float(v)) for n, q, v in zip(z["names"], z["q"], z["land"], strict=True)})
    todo = [t for t in tiles if t[0] not in done]
    log(f"{label}: {len(done)} tiles cached, {len(todo)} to read.")

    def work(tile):
        try:
            return tile[0], compute(tile)
        except Exception as exc:  # network or decoding error: skipped now, retried on the next run
            return tile[0], exc

    # next free batch number (not the file count: a fully failed batch writes no file, which left gaps)
    start = max((int(f.stem.split("_")[1]) for f in cache_dir.glob(f"{prefix}_*.npz")), default=-1) + 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for b in range(0, len(todo), batch):
            results = list(pool.map(work, todo[b:b + batch]))
            ok = [(n, r) for n, r in results if not isinstance(r, Exception)]
            if ok:
                np.savez(cache_dir / f"{prefix}_{start + b // batch:04d}.npz", names=np.array([n for n, _ in ok]),
                         q=np.stack([r[0] for _, r in ok]), land=np.array([r[1] for _, r in ok]))
                done.update({n: r for n, r in ok})
            failed = [r for _, r in results if isinstance(r, Exception)]
            log(f"  {min(b + batch, len(todo))}/{len(todo)} read, {len(failed)} failed in this batch"
                + (f" (first: {failed[0]!r})" if failed else ""))
    return done


def build_earth_tiles(limit: int | None = None, workers: int = 16, batch: int = 500,
                      cache_dir: Path | None = None, log=print) -> dict:
    """Real Earth tiles, 1 x 1 deg each: Copernicus GLO-90 topography, NASA POWER climate, bare ground (landcover.py).

    limit: process an evenly spread sample of that many tiles (quick tests).
    Returns arrays for a Dataset: tile_lat, tile_lon, valid_frac, quantiles (n, K, L).
    """
    from src import landcover, sources

    tiles = sources.copernicus_tiles()
    if limit:
        tiles = tiles[::max(1, len(tiles) // limit)][:limit]
    topo = _tile_batches(tiles, cache_dir or TOPO_CACHE_DIR, lambda t: earth_tile_topography(t[0], t[1]),
                         workers, batch, log, "Topography", "topo")
    land_tiles = [t for t in tiles if t[0] in topo and topo[t[0]][1] >= MIN_LAND_FRAC]
    worldcover = sources.worldcover_files()
    cover = _tile_batches(land_tiles, landcover.COVER_CACHE_DIR,
                          lambda t: landcover.earth_tile_bare(*t, worldcover),
                          workers, batch, log, "Bare ground", "bare")
    fields = sources.earth_climate_fields()
    lat, lon, land, quantiles, dropped = [], [], [], [], []
    order = [config.VARIABLE_NAMES.index(v)
             for v in TOPO_VARIABLES + CLIMATE_VARIABLES + ASTRO_VARIABLES + COVER_VARIABLES]
    sun: dict[int, np.ndarray] = {}                 # the Sun's elevation depends on the latitude only
    for name, lat0, lon0 in land_tiles:
        if name not in cover:
            dropped.append(name)
            continue
        if lat0 not in sun:
            sun[lat0] = sun_elevation_quantiles(lat0 + TILE_DEG / 2, "Earth")[None, :]
        q = np.empty((len(config.VARIABLE_NAMES), config.QUANTILE_LEVELS.size), dtype=np.float32)
        q[order] = np.concatenate([topo[name][0], earth_climate_quantiles(fields, lat0, lon0), sun[lat0],
                                   cover[name][0]])
        if np.isfinite(q).all():
            lat.append(lat0 + TILE_DEG / 2)
            lon.append(lon0 + TILE_DEG / 2)
            land.append(topo[name][1])
            quantiles.append(q)
        else:
            dropped.append(name)
    log(f"Earth tiles kept: {len(quantiles)} of {len(tiles)}; land tiles dropped for missing data: {len(dropped)}"
        + (f" (e.g. {dropped[:5]})" if dropped else ""))
    return {"tile_lat": np.array(lat), "tile_lon": np.array(lon), "valid_frac": np.array(land),
            "quantiles": np.stack(quantiles).astype(np.float32)}


# ===================================================== real data: planetary targets
TARGET_HALF_KM = 55.0           # target window ~110 km, the size of an Earth tile
REAL_TARGETS = (
    dict(target_id="moon_shackleton", name="Shackleton Crater region (South Pole)", body="Moon", lat=-89.67,
         lon=129.78, dem="lola_polar",
         note="IAU Gazetteer centre of Shackleton crater. The window of about 110 km holds its permanently shadowed "
              "floor, its rim and the ridge toward de Gerlache: a regional profile, not the rim alone. 'Peak near "
              "Shackleton' and 'Connecting Ridge' were among NASA's 13 Artemis III candidate regions of 2022; the "
              "list of nine of October 2024 no longer names them."),
    dict(target_id="moon_malapert", name="Malapert Massif (South Pole)", body="Moon", lat=-85.98, lon=-1.67,
         dem="lola_polar",
         note="IAU Gazetteer centre of Mons Malapert (LOLA 2011 network). 'Malapert Massif' is one of NASA's nine "
              "Artemis III candidate regions (October 2024)."),
    dict(target_id="moon_mare_tranq_pit", name="Mare Tranquillitatis pit", body="Moon", lat=8.335, lon=33.222,
         dem="lola_global", note="Pit coordinates from Robinson et al. (2012)."),
    dict(target_id="mars_jezero", name="Jezero Crater", body="Mars", lat=18.38, lon=77.58, dem="mola",
         note="IAU Gazetteer centre; Perseverance landing region."),
    dict(target_id="mars_gale", name="Gale Crater", body="Mars", lat=-5.37, lon=137.81, dem="mola",
         note="IAU Gazetteer centre; Curiosity landing region."),
    dict(target_id="mars_arcadia", name="Arcadia Planitia (ice-rich mid-latitudes)", body="Mars", lat=46.0,
         lon=-166.0, dem="mola",
         note="Approximate centre of the ice-rich zone mapped by SHARAD (Bramson et al. 2015). Future site: "
              "studied for human landings (ice as a resource), no mission selected."),
    dict(target_id="mars_oxia", name="Oxia Planum (Rosalind Franklin landing site)", body="Mars", lat=18.275,
         lon=-24.632, dem="mola",
         note="Future site of another agency: landing site of ESA's ExoMars Rosalind Franklin rover (launch planned "
              "in 2028), clay-bearing plains (Quantin-Nataf et al. 2021, Astrobiology 21(3); Fawdon et al. 2024, "
              "Journal of Maps). Coordinates 18.275 N, 335.368 E."),
)
# Heliocentric orbit and spin-axis obliquity (deg) used for the illumination of flat targets and for the Sun's
# elevation (standard values; ls_peri = solar longitude of perihelion; the Moon follows the Earth's orbit, with
# its own obliquity to the ecliptic).
ORBITS = {"Earth": dict(a_au=1.0, e=0.0167, obliquity=23.44, ls_peri=282.9),
          "Moon": dict(a_au=1.0, e=0.0167, obliquity=1.54, ls_peri=0.0),
          "Mars": dict(a_au=1.52366, e=0.0934, obliquity=25.19, ls_peri=251.0)}


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, levels: np.ndarray | None = None) -> np.ndarray:
    """Quantiles (L,) of values with non-negative weights (e.g. area or time); NaN values are skipped."""
    levels = config.QUANTILE_LEVELS if levels is None else np.asarray(levels)
    v, w = np.asarray(values, dtype=np.float64).ravel(), np.asarray(weights, dtype=np.float64).ravel()
    ok = np.isfinite(v) & (w > 0)
    if not ok.any():
        return np.full(levels.shape, np.nan, np.float32)
    order = np.argsort(v[ok])
    v, w = v[ok][order], w[ok][order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(levels, cdf, v).astype(np.float32)


def orbital_illumination(lat_deg: float, body: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-of-atmosphere daily-mean insolation (W/m2) and daylight fraction over 12 seasons of a flat site.

    Returns (insolation, daylight fraction, time weight) per season. Seasons are 30 deg
    of solar longitude; the time spent in each is proportional to r^2 (Kepler's 2nd law).
    """
    o = ORBITS[body]
    ls = np.radians(np.arange(12) * 30.0 + 15.0)
    decl = np.arcsin(np.sin(np.radians(o["obliquity"])) * np.sin(ls))
    r = o["a_au"] * (1 - o["e"] ** 2) / (1 + o["e"] * np.cos(ls - np.radians(o["ls_peri"])))
    phi = np.radians(lat_deg)
    h0 = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0))
    q = config.SOLAR_CONSTANT_WM2 / (np.pi * r ** 2) * (h0 * np.sin(phi) * np.sin(decl)
                                                         + np.cos(phi) * np.cos(decl) * np.sin(h0))
    return q, h0 / np.pi, r ** 2


def sun_elevation_samples(lat_deg: float | np.ndarray, body: str, n_seasons: int = 72,
                          n_hours: int = 288, ls_range: tuple[float, float] = (0.0, 360.0)
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Elevation of the Sun (deg) above the horizon of a smooth sphere, sampled over one year, with the time
    weight of each sample: (values, weights), flattened; samples with the Sun below the horizon weigh 0.

    lat_deg: one latitude or several (their samples are pooled, e.g. the latitudes of a target window).
    Seasons are steps of solar longitude; the time spent in each is proportional to r^2 (Kepler's 2nd law);
    the hours of a day are uniform. Astronomy only: obliquity and orbit of config-free ORBITS, no terrain,
    no atmosphere, the same definition on the three worlds.
    ls_range: solar longitudes (deg) to cover, the whole year by default.
    """
    o = ORBITS[body]
    ls = np.radians(ls_range[0] + (np.arange(n_seasons) + 0.5) * (ls_range[1] - ls_range[0]) / n_seasons)
    decl = np.arcsin(np.sin(np.radians(o["obliquity"])) * np.sin(ls))[None, :, None]
    r2 = (o["a_au"] * (1 - o["e"] ** 2) / (1 + o["e"] * np.cos(ls - np.radians(o["ls_peri"])))) ** 2
    hour = ((np.arange(n_hours) + 0.5) * 2 * np.pi / n_hours)[None, None, :]
    phi = np.radians(np.atleast_1d(np.asarray(lat_deg, dtype=np.float64)))[:, None, None]
    sin_e = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour)
    weights = np.where(sin_e > 0, r2[None, :, None], 0.0)
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0))).ravel(), weights.ravel()


def sun_elevation_quantiles(lat_deg: float | np.ndarray, body: str, levels: np.ndarray | None = None) -> np.ndarray:
    """(L,) quantiles of the Sun's elevation (deg) while it is above the horizon, over one year (see
    sun_elevation_samples)."""
    return weighted_quantiles(*sun_elevation_samples(lat_deg, body), levels)


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTH_START_DAY = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
_EQUINOX_DAY = 79.0             # 20 March: solar longitude 0 (mean motion; within ~2 deg of the true value)


def earth_monthly_sun_elevation(lat_deg: float, levels: np.ndarray | None = None) -> np.ndarray:
    """(12, L) quantiles of the Sun's elevation while it is up at an Earth latitude, month by month
    (NaN for a month of polar night). Same definition as sun_elevation_quantiles."""
    out = []
    for m in range(12):
        lo, hi = (_MONTH_START_DAY[m:m + 2] - _EQUINOX_DAY) * 360.0 / 365.25
        out.append(weighted_quantiles(*sun_elevation_samples(lat_deg, "Earth", n_seasons=10, ls_range=(lo, hi)),
                                      levels))
    return np.stack(out)


def window_latitudes(lat_deg: float, half_km: float, radius_km: float, n: int = 9) -> np.ndarray:
    """Latitudes (deg) of an n x n grid of points covering a square window of side 2 x half_km centred on a
    point of a sphere. Near a pole the window spans latitudes that differ by as much as the Sun's elevation."""
    offsets = np.linspace(-half_km, half_km, n)
    east, north = np.meshgrid(offsets, offsets)
    dist, bearing = np.hypot(east, north) / radius_km, np.arctan2(east, north)
    phi = np.radians(lat_deg)
    return np.degrees(np.arcsin(np.clip(np.sin(phi) * np.cos(dist) + np.cos(phi) * np.sin(dist) * np.cos(bearing),
                                        -1.0, 1.0))).ravel()


def horizontal_irradiance_factor(sin_elevation: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Share of the solar constant received by a horizontal surface: sin(elevation) while the Sun is visible
    and above the local horizontal plane, else 0.

    On a summit the terrain horizon lies below the horizontal plane (curvature, lower ground around), so the
    Sun can be visible at a negative elevation: the point is lit, but a horizontal surface receives nothing
    from below its own plane. Without the floor at 0 these samples counted as negative irradiance.
    """
    return np.where(visible, np.maximum(sin_elevation, 0.0), 0.0)


def polar_illumination(dem_path: str, lat: float, lon: float, half_km: float, reach_km: float = 150.0,
                       grid_km: float = 2.0, n_sun_lon: int = 72, n_decl: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Sunlit fraction and mean horizontal insolation (W/m2) of the points of a lunar polar window.

    Horizon method on LOLA topography (simplified from Mazarico et al. 2011): for each
    point, the terrain horizon is traced toward the Sun up to `reach_km` (with the Moon's
    curvature); the Sun is visible when its elevation is above that horizon. Averaged over
    a full lunar day (sub-solar longitudes) and the range of solar declination (+-1.54 deg).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform
    from rasterio.windows import from_bounds

    radius_m = BODY_RADIUS_KM["Moon"] * 1000
    geo = f"+proj=longlat +R={radius_m:.0f} +no_defs"
    with rasterio.open(dem_path) as src:
        (x0,), (y0,) = transform(geo, src.crs, [lon], [lat])
        span = (half_km + reach_km) * 1000
        n = int(round(2 * span / 1000))                                # 1 km pixels
        win = from_bounds(x0 - span, y0 - span, x0 + span, y0 + span, transform=src.transform)
        z = src.read(1, window=win, out_shape=(n, n), resampling=Resampling.average).astype(np.float64) * 1000
        sun_lon = np.arange(n_sun_lon) * 360.0 / n_sun_lon
        px, py = transform(geo, src.crs, [0.0] + list(sun_lon), [-90.0] + [-80.0] * n_sun_lon)
        offsets = np.arange(-half_km, half_km + 1e-9, grid_km) * 1000
        gx, gy = np.meshgrid(x0 + offsets, y0 + offsets)
        plon, plat = transform(src.crs, geo, gx.ravel().tolist(), gy.ravel().tolist())
    pix = 2 * span / n
    u = np.column_stack([np.subtract(px[1:], px[0]), np.subtract(py[1:], py[0])])
    u /= np.linalg.norm(u, axis=1, keepdims=True)                        # map direction toward each sub-solar lon

    def rc(x, y):
        return ((y0 + span) - y) / pix - 0.5, (x - (x0 - span)) / pix - 0.5

    pts_x, pts_y = gx.ravel(), gy.ravel()
    z_pt = ndimage.map_coordinates(z, rc(pts_x, pts_y), order=1, mode="nearest")
    d = np.arange(1, int(reach_km) + 1) * 1000.0
    horizon = np.empty((n_sun_lon, pts_x.size))
    for k, (ux, uy) in enumerate(u):
        zs = ndimage.map_coordinates(z, rc(pts_x[:, None] + ux * d, pts_y[:, None] + uy * d), order=1, mode="nearest")
        horizon[k] = np.arctan((zs - z_pt[:, None] - d ** 2 / (2 * radius_m)) / d).max(axis=1)
    decl = np.radians(ORBITS["Moon"]["obliquity"]) * np.sin(np.linspace(0, 2 * np.pi, n_decl, endpoint=False))
    phi, lam = np.radians(np.asarray(plat)), np.radians(np.asarray(plon))
    lit = np.zeros(pts_x.size)
    flux = np.zeros(pts_x.size)
    for k, s in enumerate(np.radians(sun_lon)):
        for dj in decl:
            sin_e = np.sin(phi) * np.sin(dj) + np.cos(phi) * np.cos(dj) * np.cos(lam - s)
            visible = np.arcsin(np.clip(sin_e, -1, 1)) > horizon[k]
            lit += visible
            flux += horizontal_irradiance_factor(sin_e, visible)
    n_samples = n_sun_lon * n_decl
    return lit / n_samples, config.SOLAR_CONSTANT_WM2 * flux / n_samples


def diviner_thermal(cells: np.ndarray, lat: float, lon: float, half_km: float, min_bins: int = 24,
                    levels: np.ndarray | None = None) -> np.ndarray:
    """(2, L) diurnal amplitude and minimum of Diviner Tbol over the 0.5 deg cells of a window (area-weighted).

    cells: rows (lon, lat, local time, Tbol), 96 local-time bins per cell in a row.
    """
    block = cells.reshape(-1, 96, 4)
    c_lon, c_lat, tb = block[:, 0, 0], block[:, 0, 1], block[:, :, 3]
    inside = math_engine.haversine_km(lat, lon, c_lat, c_lon, BODY_RADIUS_KM["Moon"]) <= half_km
    enough = np.isfinite(tb).sum(axis=1) >= min_bins
    keep = inside & enough
    tmax, tmin = np.nanmax(tb[keep], axis=1), np.nanmin(tb[keep], axis=1)
    w = np.cos(np.radians(c_lat[keep]))
    return np.stack([weighted_quantiles(tmax - tmin, w, levels), weighted_quantiles(tmin, w, levels)])


MARS_SEASONS_LS = np.arange(12) * 30.0 + 15.0    # the 12 seasons of orbital_illumination (solar longitude, deg)


def diurnal_cycle_thermal(cycles: np.ndarray, weights: np.ndarray, levels: np.ndarray | None = None) -> np.ndarray:
    """(2, L) diurnal amplitude and minimum (K) of the surface temperature over the year.

    cycles: (S, H) mean diurnal cycle of each season; weights: (S,) time spent in each season.
    Amplitude = max - min of the cycle, the counterpart of the Earth variable (mean daily max - mean daily
    min); minimum = coldest hour of the mean cycle.
    """
    cycles = np.asarray(cycles, dtype=np.float64)
    low = cycles.min(axis=1)
    return np.stack([weighted_quantiles(cycles.max(axis=1) - low, weights, levels),
                     weighted_quantiles(low, weights, levels)])


def _in_situ_control(target_id: str, variable: str, values: np.ndarray) -> str:
    """Sentence comparing model values with the published in-situ range of config.TARGET_REFERENCE_VALUES."""
    ref = config.TARGET_REFERENCE_VALUES.get(target_id, {}).get(variable)
    if ref is None:
        return ""
    return (f" In-situ control: {ref[0]:.0f}-{ref[1]:.0f} K ({ref[2]}); model range over the year "
            f"{np.min(values):.0f}-{np.max(values):.0f} K.")


def _range_quantiles(low: float, high: float, levels: np.ndarray | None = None) -> np.ndarray:
    """Quantiles of a value known only as a (low, high) range: uniform spread."""
    return sample_quantiles(np.linspace(low, high, 101), levels)


def read_target_dem(target: Mapping, polar_path: str | None = None) -> tuple[np.ndarray, tuple, str]:
    """Elevation window (m) of a planetary target at the common ~463 m pixel: (dem, (dy, dx), source label)."""
    from src import sources

    lat, lon, radius = target["lat"], target["lon"], BODY_RADIUS_KM[target["body"]]
    if target["dem"] == "mola":
        path, projection, unit, label = (sources.mola_tile(lat, lon % 360), "equirectangular", 1.0,
                                         "MGS MOLA MEGDR 128 ppd, ~463 m (PDS)")
    elif target["dem"] == "lola_polar":
        path, projection, unit, label = (polar_path or str(sources.lola_polar_local()), "polar", 1000.0,
                                         "LRO LOLA polar DEM, 240 m (PDS)")
    else:
        path, projection, unit, label = (sources.LOLA_GLOBAL + ".xml", "equirectangular", 1000.0,
                                         "LRO LOLA LDEM_64, ~474 m (PDS)")
    z, size = sources.read_dem_window(path, lat, lon, TARGET_HALF_KM, radius, TOPO_PIXEL_M, projection, unit)
    return z, size, label


def build_real_targets(log=print) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Profiles of the planetary targets from PDS data, physics and published values.

    Returns (targets table, quantiles (T, K, L), {target_id: {variable: source text}}).
    """
    from src import sources

    idx = {v: i for i, v in enumerate(config.VARIABLE_NAMES)}
    n_lev = config.QUANTILE_LEVELS.size
    polar = str(sources.lola_polar_local())
    rows, quantiles, provenance = [], [], {}
    for t in REAL_TARGETS:
        q = np.full((len(idx), n_lev), np.nan, dtype=np.float32)
        src: dict[str, str] = {}
        body, lat, lon = t["body"], t["lat"], t["lon"]
        radius = BODY_RADIUS_KM[body]
        # --- topography
        z, size, dem_label = read_target_dem(t, polar)
        q[[idx["slope_deg"], idx["roughness_m"]]] = topography_quantiles(z, size)
        src["slope_deg"] = src["roughness_m"] = dem_label
        # --- thermal
        if body == "Moon":
            band = "90s80s" if lat < -80 else "00n10n" if 0 <= lat < 10 else None
            if band:
                q[[idx["diurnal_amp_k"], idx["temp_min_k"]]] = diviner_thermal(
                    sources.diviner_cells(band), lat, lon, TARGET_HALF_KM)
                src["diurnal_amp_k"] = src["temp_min_k"] = "LRO Diviner Global Cumulative Product, Tbol (PDS)"
        if body == "Mars":   # Mars Climate Database: mean diurnal cycle at the target centre, 12 seasons
            cycles = np.stack([sources.mcd_surface_temperature(lat, lon, ls) for ls in MARS_SEASONS_LS])
            q[[idx["diurnal_amp_k"], idx["temp_min_k"]]] = diurnal_cycle_thermal(
                cycles, orbital_illumination(lat, body)[2])
            low = cycles.min(axis=1)
            for var, values in (("diurnal_amp_k", cycles.max(axis=1) - low), ("temp_min_k", low)):
                src[var] = (f"{sources.MCD_CITATION}: surface temperature at the target centre, 12 seasons x 24 "
                            "local times." + _in_situ_control(t["target_id"], var, values))
        for var, (low, high, text) in config.BODY_REFERENCE_VALUES.get(body, {}).items():
            q[idx[var]] = _range_quantiles(low, high)
            src[var] = text
        # --- illumination
        if t["dem"] == "lola_polar":
            sunlit, insol = polar_illumination(polar, lat, lon, TARGET_HALF_KM)
            q[idx["insolation_wm2"]] = sample_quantiles(insol)
            src["insolation_wm2"] = "Computed: horizon method on LRO LOLA topography"
            official, _ = sources.read_dem_window(sources.PGDA_ILLUMINATION_SOUTH, lat, lon, TARGET_HALF_KM, radius,
                                                  TOPO_PIXEL_M, "polar", scale_from_file=True)
            q_off, q_ours = sample_quantiles(official), sample_quantiles(sunlit)
            if np.isfinite(official).mean() > 0.9:
                q[idx["sunlit_frac"]] = q_off
                src["sunlit_frac"] = (f"NASA PGDA average illumination map (Mazarico et al. 2011), 120 m; our horizon "
                                      f"method on LOLA agrees within W1 = {np.mean(np.abs(q_off - q_ours)):.3f}")
            else:
                q[idx["sunlit_frac"]] = q_ours
                src["sunlit_frac"] = "Computed: horizon method on LRO LOLA topography"
            log(f"  {t['target_id']} sunlit fraction median: official {q_off[9]:.2f}, ours {q_ours[9]:.2f}; "
                f"p90 official {q_off[16]:.2f}, ours {q_ours[16]:.2f}")
        else:
            insol, day, w = orbital_illumination(lat, body)
            transmission = config.MARS_ATMOSPHERE_TRANSMISSION if body == "Mars" else 1.0
            q[idx["insolation_wm2"]] = weighted_quantiles(insol * transmission, w)
            q[idx["sunlit_frac"]] = weighted_quantiles(day, w)
            src["insolation_wm2"] = ("Computed: orbit and obliquity" +
                                     (f", atmospheric transmission {transmission}" if body == "Mars" else ""))
            src["sunlit_frac"] = "Computed: astronomical daylight fraction" + (", no clouds" if body == "Mars" else "")
        # --- height of the Sun: astronomy over the latitudes of the window, same definition as on Earth
        q[idx["sun_elevation_deg"]] = sun_elevation_quantiles(window_latitudes(lat, TARGET_HALF_KM, radius), body)
        src["sun_elevation_deg"] = (f"Computed: astronomy (latitudes of the window, axial tilt "
                                    f"{ORBITS[body]['obliquity']} deg), smooth sphere, terrain shadows not included")
        missing = [v for v in config.VARIABLE_NAMES if v not in src]
        log(f"Target {t['target_id']}: measured {len(src)}/{len(idx)}"
            + (f", not evaluated: {missing}" if missing else ""))
        rows.append({k: t[k] for k in ("target_id", "name", "body", "lat", "lon", "note")})
        quantiles.append(q)
        provenance[t["target_id"]] = src
    return pd.DataFrame(rows), np.stack(quantiles), provenance


def tile_area_weights(ds_or_lat, land_frac: np.ndarray | None = None) -> np.ndarray:
    """Share of Earth's land area held by each tile of a dataset (math_engine.land_area_weights)."""
    if isinstance(ds_or_lat, Dataset):
        return math_engine.land_area_weights(ds_or_lat.tile_lat, ds_or_lat.valid_frac)
    return math_engine.land_area_weights(ds_or_lat, land_frac)


REAL_SOURCES = {"Earth topography": "Copernicus DEM GLO-90 (ESA), AWS open data",
                "Earth climate": "NASA POWER climatology (MERRA-2, CERES SYN1deg), AWS open data",
                "Earth bare ground": "ESA WorldCover 10 m 2021 north of 60 S; SCAR Antarctic Digital Database "
                                     "rock outcrop v7.11 south of it (both CC BY 4.0)",
                "Moon": "NASA PDS: LRO LOLA, LRO Diviner; NASA PGDA illumination",
                "Mars": "NASA PDS: MGS MOLA; Mars Climate Database v6.2 (LMD, with ESA and CNES)"}


def rebuild_real_targets(ds: Dataset, workers: int = 8, log=print) -> None:
    """Recompute, in place, the target profiles of a real dataset (and the layers that depend on the ranking).

    The Earth tiles are left untouched; the default tolerances, which derive from them, are recomputed.
    """
    ds.sigma_default = math_engine.default_tolerances(ds.quantiles, weights=tile_area_weights(ds))
    targets, target_q, provenance = build_real_targets(log=log)
    if targets["target_id"].tolist() != ds.targets["target_id"].tolist():
        raise ValueError("the target list changed: rebuild the whole dataset with --real")
    ds.targets, ds.target_quantiles = targets, target_q
    ds.meta.update({"sources": dict(REAL_SOURCES), "target_sources": provenance,
                    "targets_rebuilt": datetime.now(UTC).isoformat(timespec="seconds")})
    ds.validate()
    build_real_layers(ds, workers=workers, log=log)


def build_real_dataset(limit: int | None = None, workers: int = 16, log=print) -> Dataset:
    """Real dataset: Earth tiles (Copernicus DEM + NASA POWER) and planetary targets (PDS)."""
    earth = build_earth_tiles(limit=limit, workers=workers, log=log)
    targets, target_q, provenance = build_real_targets(log=log)
    n = earth["quantiles"].shape[0]
    known = known_analogs_table()
    ds = Dataset(
        variables=config.VARIABLE_NAMES, levels=config.QUANTILE_LEVELS.copy(), tile_id=np.arange(n),
        tile_lat=earth["tile_lat"], tile_lon=earth["tile_lon"], valid_frac=earth["valid_frac"],
        quantiles=earth["quantiles"],
        sigma_default=math_engine.default_tolerances(
            earth["quantiles"], weights=tile_area_weights(earth["tile_lat"], earth["valid_frac"])),
        targets=targets, target_quantiles=target_q, known_analogs=known,
        meta={"source": "real", "version": 1, "n_tiles": int(n), "sample_limit": limit,
              "created": datetime.now(UTC).isoformat(timespec="seconds"),
              "resolution_note": "Earth tiles of 1 x 1 deg, weighted by their land area; topography at a common "
                                 "~463 m pixel on every world; Earth climate on the NASA POWER grids (0.5 x 0.625 "
                                 "deg and 1 deg), 12 months.",
              "sources": dict(REAL_SOURCES), "target_sources": provenance})
    ds.validate()
    build_real_layers(ds, workers=min(workers, 8), log=log)
    return ds


# ================================================= fine-scale zoom
ZOOM_BLOCK_PX = 24              # a sub-area is about 24 x 24 pixels of ~463 m: ~11 km on the ground
ZOOM_MIN_LAND_FRAC = 0.5        # a sub-area needs at least this share of land pixels
ZOOM_FILE = "zoom.npz"
_DEM_NO_DATA = np.iinfo(np.int16).min


@dataclass
class TileZoom:
    """Fine-scale view of one Earth tile: its DEM and the terrain statistics of its ~11 km sub-areas."""

    lat0: int                   # south-west corner of the 1 x 1 deg tile
    lon0: int
    dem: np.ndarray             # (rows, cols) elevation in metres, north-up, NaN = water
    quantiles: np.ndarray       # (S, 2, L) slope and roughness quantiles per sub-area (NaN = not enough land)
    land: np.ndarray            # (S,) land fraction per sub-area
    grid: tuple[int, int]       # sub-area rows and columns; S = rows * cols, north to south then west to east

    def bounds(self) -> np.ndarray:
        """(S, 4) south, north, west, east of every sub-area, in degrees."""
        r = block_edges(self.dem.shape[0], n_blocks=self.grid[0]) / self.dem.shape[0]
        c = block_edges(self.dem.shape[1], n_blocks=self.grid[1]) / self.dem.shape[1]
        north, south = self.lat0 + 1 - r[:-1], self.lat0 + 1 - r[1:]
        west, east = self.lon0 + c[:-1], self.lon0 + c[1:]
        return np.array([(south[i], north[i], west[j], east[j])
                         for i in range(self.grid[0]) for j in range(self.grid[1])])

    def pixel_size_m(self) -> tuple[float, np.ndarray]:
        """(dy, dx per row) of the DEM pixels in metres."""
        rows, cols = self.dem.shape
        rows_lat = self.lat0 + 1 - (np.arange(rows) + 0.5) / rows
        return geographic_pixel_size(rows_lat, 1 / rows, 1 / cols, BODY_RADIUS_KM["Earth"])


def zoom_key(lat: float, lon: float) -> tuple[int, int]:
    """Key of the 1 x 1 deg tile containing a point: its south-west corner (lat0, lon0)."""
    return int(np.floor(lat)), int(np.floor(lon))


def block_edges(n_px: int, block_px: int = ZOOM_BLOCK_PX, n_blocks: int | None = None) -> np.ndarray:
    """Pixel edges of the blocks that cover n_px pixels with blocks of about block_px (at least one)."""
    n_blocks = max(1, int(round(n_px / block_px))) if n_blocks is None else n_blocks
    return np.linspace(0, n_px, n_blocks + 1).astype(int)


def subarea_topography(dem_m: np.ndarray, pixel_size_m: float | tuple, block_px: int = ZOOM_BLOCK_PX,
                       levels: np.ndarray | None = None, min_land_frac: float = ZOOM_MIN_LAND_FRAC
                       ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Slope/roughness quantiles of every ~block_px sub-area of a DEM: ((S, 2, L), land (S,), (rows, cols)).

    Slope and roughness are computed on the whole DEM first, with the operators and the pixel
    of the global screening, then sampled per sub-area: only the sampling window shrinks.
    Sub-areas with less than `min_land_frac` land get NaN quantiles.
    """
    dem = np.asarray(dem_m)
    slope = slope_degrees(dem, pixel_size_m)
    rough = local_roughness(dem, ROUGHNESS_WINDOW_PX)
    r_edges, c_edges = block_edges(dem.shape[0], block_px), block_edges(dem.shape[1], block_px)
    quantiles, land = [], []
    for r0, r1 in zip(r_edges[:-1], r_edges[1:], strict=True):
        for c0, c1 in zip(c_edges[:-1], c_edges[1:], strict=True):
            frac = float(np.isfinite(dem[r0:r1, c0:c1]).mean())
            q = np.stack([sample_quantiles(slope[r0:r1, c0:c1], levels), sample_quantiles(rough[r0:r1, c0:c1], levels)])
            quantiles.append(q if frac >= min_land_frac else np.full_like(q, np.nan))
            land.append(frac)
    return np.stack(quantiles), np.array(land, dtype=np.float32), (len(r_edges) - 1, len(c_edges) - 1)


def default_zoom_tiles(ds: Dataset, n_sites: int = config.TOP_SITES_SHOWN) -> list[tuple[int, int]]:
    """Tiles worth a zoom, as sorted (lat0, lon0) keys: for every target and mission profile, the distinct
    sites the app shows with its default settings, plus the best under-studied site."""
    from src import analysis

    ctx = analysis.build_context(ds)
    keys: set[tuple[int, int]] = set()
    for t in range(len(ds.targets)):
        w1 = math_engine.wasserstein1(ds.quantiles, ds.target_quantiles[t])
        reach = analysis.target_reachability(ctx, w1)
        for mission in config.MISSION_PROFILES:
            sites = analysis.analyse_target(ctx, t, analysis.mission_dimension_weights(mission),
                                            w1=w1, reach=reach).ranking.site_idx
            picks = [int(i) for i in sites[:n_sites]]
            best_new = analysis.best_understudied(ctx, sites)
            if best_new is not None:
                picks.append(best_new)
            keys.update(zoom_key(ds.tile_lat[i], ds.tile_lon[i]) for i in picks)
    return sorted(keys)


def build_zoom(ds: Dataset, workers: int = 8, log=print) -> tuple[dict, dict]:
    """Fine-scale zoom of the real dataset: ({(lat0, lon0): TileZoom}, {target_id: elevation window (m)}).

    Reads again the Copernicus DEM of the tiles of `default_zoom_tiles` (a few hundred kB each).
    """
    from concurrent.futures import ThreadPoolExecutor

    from src import sources

    names = {(lat, lon): name for name, lat, lon in sources.copernicus_tiles()}
    keys = [k for k in default_zoom_tiles(ds) if k in names]

    def work(key):
        try:
            dem, size = earth_tile_dem(names[key], key[0])
            quantiles, land, grid = subarea_topography(dem, size)
            return key, TileZoom(key[0], key[1], dem, quantiles, land, grid)
        except Exception as exc:  # network or decoding error: retried once below
            return key, exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(pool.map(work, keys))
    for key in [k for k, z in results.items() if isinstance(z, Exception)]:
        results[key] = work(key)[1]
    zoom = {k: z for k, z in results.items() if isinstance(z, TileZoom)}
    log(f"Zoom: {len(zoom)} of {len(keys)} tiles read, {len(keys) - len(zoom)} failed.")
    by_id = {t["target_id"]: t for t in REAL_TARGETS}
    target_dems = {tid: read_target_dem(by_id[tid])[0] for tid in ds.targets["target_id"] if tid in by_id}
    return zoom, target_dems


def zoom_meta(zoom: Mapping) -> dict:
    """What meta.json says about the zoom."""
    return {"tiles": len(zoom), "block_px": ZOOM_BLOCK_PX, "pixel_m": TOPO_PIXEL_M,
            "note": "Sub-areas of about 11 km inside the sites shown with the default settings of each mission "
                    "profile; same DEM, pixel and operators as the screening; climate stays at tile scale."}


def _pack_dem(dem: np.ndarray) -> np.ndarray:
    """Elevation as int16 metres (NaN -> no-data): enough for display, a quarter of the float size."""
    return np.where(np.isfinite(dem), np.round(dem), _DEM_NO_DATA).astype(np.int16)


def _unpack_dem(packed: np.ndarray) -> np.ndarray:
    return np.where(packed == _DEM_NO_DATA, np.nan, packed).astype(np.float32)


def save_zoom(directory: str | os.PathLike, zoom: Mapping, target_dems: Mapping) -> Path:
    """Write zoom.npz: keys (m, 2), then dem_i / q_i / land_i / grid_i per tile and target_<id> per target."""
    keys = sorted(zoom)
    arrays = {"keys": np.array(keys, dtype=np.int16).reshape(len(keys), 2)}
    for i, key in enumerate(keys):
        z = zoom[key]
        arrays.update({f"dem_{i}": _pack_dem(z.dem), f"q_{i}": z.quantiles.astype(np.float32),
                       f"land_{i}": z.land.astype(np.float32), f"grid_{i}": np.array(z.grid, dtype=np.int16)})
    arrays.update({f"target_{tid}": _pack_dem(dem) for tid, dem in target_dems.items()})
    path = Path(directory) / ZOOM_FILE
    np.savez_compressed(path, **arrays)
    return path


def load_zoom(directory: str | os.PathLike) -> tuple[dict, dict]:
    """Read zoom.npz written by `save_zoom`; ({}, {}) when the dataset has none."""
    path = Path(directory) / ZOOM_FILE
    if not path.is_file():
        return {}, {}
    zoom, target_dems = {}, {}
    with np.load(path) as z:
        for i, (lat0, lon0) in enumerate(z["keys"].tolist()):
            zoom[(lat0, lon0)] = TileZoom(lat0, lon0, _unpack_dem(z[f"dem_{i}"]), z[f"q_{i}"], z[f"land_{i}"],
                                          tuple(int(v) for v in z[f"grid_{i}"]))
        target_dems = {name.removeprefix("target_"): _unpack_dem(z[name])
                       for name in z.files if name.startswith("target_")}
    return zoom, target_dems


# ============================================ subsurface case studies
SUBSURFACE_FILE = "subsurface.csv"
SUBSURFACE_COLUMNS = ("name", "country", "lat", "lon", "maat_k", "source")


def build_subsurface() -> pd.DataFrame:
    """Earth lava-tube fields of config.LAVA_TUBE_SITES with the mean annual air temperature (K) of their
    NASA POWER cell (T2M, MERRA-2, 0.5 x 0.625 deg): a cell average, not the temperature at the cave."""
    from src import sources

    grid_lat, grid_lon = sources.POWER_STORES["merra2"][1:]
    annual_c = sources.power_climatology("merra2", "T2M")[12]
    rows = []
    for name, country, lat, lon in config.LAVA_TUBE_SITES:
        cell = annual_c[np.abs(grid_lat - lat).argmin(), np.abs(grid_lon - lon).argmin()]
        rows.append((name, country, lat, lon, round(float(cell) + 273.15, 2),
                     f"{config.LAVA_TUBE_SOURCE}. Temperature: NASA POWER climatology (T2M, annual mean)"))
    return pd.DataFrame(rows, columns=list(SUBSURFACE_COLUMNS))


def save_layers(ds: Dataset, directory: str | os.PathLike) -> None:
    """Write the optional layers of a dataset (fine-scale zoom, subsurface case studies, place names) next to it."""
    from src import places

    if ds.zoom:
        save_zoom(directory, ds.zoom, ds.target_dems)
    if ds.subsurface is not None:
        ds.subsurface.to_csv(Path(directory) / SUBSURFACE_FILE, index=False)
    if ds.places is not None:
        ds.places.to_csv(Path(directory) / places.PLACES_FILE, index=False, lineterminator="\n")


def load_layers(ds: Dataset, directory: str | os.PathLike) -> None:
    """Read the optional layers written by `save_layers`, when they exist."""
    from src import places

    ds.zoom, ds.target_dems = load_zoom(directory)
    path = Path(directory) / SUBSURFACE_FILE
    ds.subsurface = pd.read_csv(path) if path.is_file() else None
    path = Path(directory) / places.PLACES_FILE
    ds.places = None
    if path.is_file():
        table = pd.read_csv(path, keep_default_na=False)
        # one row per tile, in tile order: a file from another build is ignored rather than misread
        same_tiles = len(table) == ds.n_tiles
        if same_tiles and np.allclose(table["lat"], ds.tile_lat) and np.allclose(table["lon"], ds.tile_lon):
            ds.places = table


def build_place_layer(ds: Dataset, log=print) -> None:
    """Name every tile (country, physical region) from Natural Earth, in place; display only (src/places.py)."""
    from src import places

    ds.places = places.build_places(ds.tile_lat, ds.tile_lon, log=log)
    ds.meta["places"] = {"source": places.NATURAL_EARTH_CITATION,
                         "note": "Country and named physical region of each tile centre; display only, never "
                                 "part of the score. Country lines are Natural Earth's de facto boundaries; no "
                                 "country is given in the areas it lists as disputed."}


def build_real_layers(ds: Dataset, workers: int = 8, log=print) -> None:
    """Compute the optional layers of a real dataset in place and describe them in its meta."""
    ds.zoom, ds.target_dems = build_zoom(ds, workers=workers, log=log)
    ds.subsurface = build_subsurface()
    ds.meta["zoom"] = zoom_meta(ds.zoom)
    ds.meta["subsurface"] = {"sites": len(ds.subsurface),
                             "note": "Lava-tube case studies; temperature = annual mean of the NASA POWER cell."}
    build_place_layer(ds, log=log)


# ===================================================================== I/O
def save_dataset(ds: Dataset, directory: str | os.PathLike) -> Path:
    """Write a dataset following the data contract (compressed npz + csv + json)."""
    ds.validate()
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "tiles.npz", tile_id=ds.tile_id, lat=ds.tile_lat, lon=ds.tile_lon, valid_frac=ds.valid_frac,
        quantiles=ds.quantiles, levels=ds.levels, sigma_default=ds.sigma_default,
        target_quantiles=ds.target_quantiles)
    ds.targets.to_csv(out / "targets.csv", index=False)
    ds.known_analogs.to_csv(out / "known_analogs.csv", index=False)
    meta = {**ds.meta, "variables": list(ds.variables)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_layers(ds, out)
    return out


def load_dataset(directory: str | os.PathLike) -> Dataset:
    """Read a dataset written by `save_dataset` (or by the real pipeline) and validate it."""
    d = Path(directory)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    with np.load(d / "tiles.npz") as z:
        ds = Dataset(
            variables=tuple(meta["variables"]), levels=z["levels"], tile_id=z["tile_id"], tile_lat=z["lat"],
            tile_lon=z["lon"], valid_frac=z["valid_frac"], quantiles=z["quantiles"],
            sigma_default=z["sigma_default"], targets=pd.read_csv(d / "targets.csv"),
            target_quantiles=z["target_quantiles"], known_analogs=pd.read_csv(d / "known_analogs.csv"),
            meta=meta)
    ds.validate()
    load_layers(ds, d)
    return ds


def known_analogs_table() -> pd.DataFrame:
    """The sourced catalogue of config.KNOWN_ANALOGS in the data-contract format (known_analogs.csv).

    tested_mars / tested_moon: documented properties (variable or dimension names) separated by '|'
    ('' = not used by the credibility check).
    """
    return pd.DataFrame([{
        "name": a.name, "lat": a.lat, "lon": a.lon, "relevant_to": a.bodies,
        "tested_mars": "|".join(a.mars), "tested_moon": "|".join(a.moon),
        "extent_km": a.extent_km, "source": a.source} for a in config.KNOWN_ANALOGS])


def analog_properties(known_analogs: pd.DataFrame, body: str) -> list[tuple[str, ...]]:
    """Per known analog, the documented properties it is tested on for `body` (variable or dimension
    names; empty tuple = not tested). Without the column, every dimension of the relevant sites."""
    col = f"tested_{body.lower()}"
    if col not in known_analogs.columns:
        mask = analogs_relevant_to(known_analogs, body)
        return [tuple(config.DIMENSIONS) if m else () for m in mask]
    return [tuple(s for s in str(v).split("|") if s) for v in known_analogs[col].fillna("")]


def property_weights(items: tuple[str, ...], usable: np.ndarray | None = None) -> np.ndarray:
    """(K,) weights testing a site on its documented properties: each property (a variable, or a
    dimension spread over its variables) weighs the same; unusable variables are dropped.

    Raises ValueError on a name that is neither a variable nor a dimension.
    """
    var_dims = np.asarray(config.VAR_DIMENSIONS)
    usable = np.ones(var_dims.size, dtype=bool) if usable is None else np.asarray(usable, dtype=bool)
    w = np.zeros(var_dims.size)
    for item in items:
        if item in config.VARIABLE_NAMES:
            cols = np.array([config.VARIABLE_NAMES.index(item)])
        elif item in config.DIMENSIONS:
            cols = np.flatnonzero(var_dims == item)
        else:
            raise ValueError(f"unknown property {item!r}: use a variable or a dimension name")
        cols = cols[usable[cols]]
        if cols.size:
            w[cols] += 1.0 / cols.size
    return w / w.sum() if w.sum() > 0 else w


def analogs_relevant_to(known_analogs: pd.DataFrame, body: str) -> np.ndarray:
    """Boolean mask of the known analogs used by the credibility check for `body`.

    Uses the tested dimensions when the catalogue has them, else `relevant_to`
    (all True if the catalogue has neither).
    """
    col = f"tested_{body.lower()}"
    if col in known_analogs.columns:
        return (known_analogs[col].fillna("").astype(str).str.len() > 0).to_numpy(dtype=bool)
    if "relevant_to" not in known_analogs.columns:
        return np.ones(len(known_analogs), dtype=bool)
    tags = known_analogs["relevant_to"].fillna("").astype(str).str.split(";")
    return tags.apply(lambda items: body in (s.strip() for s in items)).to_numpy(dtype=bool)


def find_dataset_dir() -> Path | None:
    """The dataset the app and the atlas load: $ANALOG_DATASET_DIR when set, else data/processed/real_v1;
    None when it holds no meta.json. No scan of data/processed/: a sample (real_sample, from --real --limit)
    is loaded only when named, and a named dataset that is missing is never replaced by another."""
    env = os.environ.get(DATASET_ENV_VAR)
    directory = Path(env) if env else DEFAULT_DATASET_DIR
    return directory if (directory / "meta.json").is_file() else None


def load_default_dataset() -> Dataset:
    """Load the dataset of find_dataset_dir. No fallback: without it the app and the atlas stop."""
    found = find_dataset_dir()
    if found is None:
        named = os.environ.get(DATASET_ENV_VAR)
        raise FileNotFoundError(f"no dataset in {named or DEFAULT_DATASET_DIR}"
                                + (f" (set by ${DATASET_ENV_VAR})" if named else "")
                                + "; build it with `python -m src.data_prep --real`")
    return load_dataset(found)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the real dataset in the final data format.")
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--real", action="store_true", help="real dataset (downloads to data/raw/, ~1 h the first time)")
    kind.add_argument("--targets", action="store_true",
                      help="recompute the target profiles (and the layers) of an existing real dataset; the Earth "
                           "tiles are kept (a few minutes)")
    kind.add_argument("--layers", action="store_true",
                      help="add the optional layers (fine-scale zoom, subsurface case studies) to an existing "
                           "real dataset, without rebuilding it (about a minute)")
    kind.add_argument("--places", action="store_true",
                      help="add the place names (country, region; Natural Earth) to an existing real dataset; "
                           "nothing else is touched (a few seconds)")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--limit", type=int, default=None, help="real: evenly spread sample of N Earth tiles")
    parser.add_argument("--workers", type=int, default=16, help="real: parallel downloads")
    args = parser.parse_args(argv)
    if args.targets:
        out = Path(args.out or DEFAULT_DATASET_DIR)
        ds = load_dataset(out)
        rebuild_real_targets(ds, workers=min(args.workers, 8))
        print(f"Targets rebuilt: {len(ds.targets)} targets, zoom of {len(ds.zoom)} tiles -> {save_dataset(ds, out)}")
        return
    if args.layers:
        out = Path(args.out or DEFAULT_DATASET_DIR)
        ds = load_dataset(out)
        build_real_layers(ds, workers=min(args.workers, 8))
        (out / "meta.json").write_text(json.dumps({**ds.meta, "variables": list(ds.variables)}, indent=2),
                                       encoding="utf-8")
        save_layers(ds, out)
        print(f"Layers: zoom of {len(ds.zoom)} tiles, {len(ds.subsurface)} subsurface case studies -> {out}")
        return
    if args.places:
        out = Path(args.out or DEFAULT_DATASET_DIR)
        ds = load_dataset(out)
        build_place_layer(ds)
        (out / "meta.json").write_text(json.dumps({**ds.meta, "variables": list(ds.variables)}, indent=2),
                                       encoding="utf-8")
        save_layers(ds, out)
        print(f"Places: {len(ds.places)} tiles named -> {out}")
        return
    ds = build_real_dataset(limit=args.limit, workers=args.workers)
    out = args.out or ROOT / "data" / "processed" / ("real_sample" if args.limit else "real_v1")
    path = save_dataset(ds, out)
    print(f"{ds.meta['source'].capitalize()} dataset: {ds.n_tiles} tiles, {len(ds.targets)} targets -> {path}")


if __name__ == "__main__":
    main()
