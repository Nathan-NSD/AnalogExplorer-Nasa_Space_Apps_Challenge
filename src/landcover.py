"""Earth bare ground, the surface-cover variable.

Share of every ~463 m pixel of an Earth tile that is bare rock or soil: ESA WorldCover 10 m north of
60 S, SCAR ADD rock outcrop polygons south of it. Used by data_prep.build_earth_tiles; no Streamlit.
"""
from __future__ import annotations

import threading

import numpy as np

from src import data_prep

COVER_SUBSAMPLE = 5             # land-cover samples per ~463 m pixel side: the bare share of a pixel is a mean of 25
COVER_CACHE_DIR = data_prep.ROOT / "data" / "processed" / "cache" / "bare_ground"
ANTARCTIC_LAT = -60             # WorldCover stops at 60 S; the SCAR ADD rock outcrop layer covers Antarctica
WORLDCOVER_BARE = (60,)         # "Bare / sparse vegetation": less than 10 % vegetation cover (WorldCover v200 PUM)
WORLDCOVER_NO_LAND = (0, 80)    # no data, permanent water bodies
# EPSG:3031, Antarctic Polar Stereographic: WGS 84 ellipsoid, latitude of true scale 71 S, central meridian 0.
_WGS84_A, _WGS84_F, _PS_LAT_TS = 6378137.0, 1 / 298.257223563, -71.0


def polar_stereographic_south_to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse EPSG:3031 projection (Snyder 1987, Map Projections - A Working Manual, eqs. 21-33 to 21-40,
    with the south-pole sign changes): metres -> (longitude, latitude) in degrees."""
    e2 = _WGS84_F * (2 - _WGS84_F)
    e = np.sqrt(e2)
    phi_c = np.radians(-_PS_LAT_TS)                                   # north-pole form: true scale at 71 N
    m_c = np.cos(phi_c) / np.sqrt(1 - e2 * np.sin(phi_c) ** 2)
    t_c = np.tan(np.pi / 4 - phi_c / 2) / ((1 - e * np.sin(phi_c)) / (1 + e * np.sin(phi_c))) ** (e / 2)
    xs, ys = -np.asarray(x, dtype=np.float64), -np.asarray(y, dtype=np.float64)
    t = np.hypot(xs, ys) * t_c / (_WGS84_A * m_c)
    chi = np.pi / 2 - 2 * np.arctan(t)
    phi = (chi + (e2 / 2 + 5 * e2**2 / 24 + e2**3 / 12 + 13 * e2**4 / 360) * np.sin(2 * chi)
           + (7 * e2**2 / 48 + 29 * e2**3 / 240 + 811 * e2**4 / 11520) * np.sin(4 * chi)
           + (7 * e2**3 / 120 + 81 * e2**4 / 1120) * np.sin(6 * chi)
           + (4279 * e2**4 / 161280) * np.sin(8 * chi))
    lam = np.arctan2(xs, -ys)
    return -np.degrees(lam), -np.degrees(phi)


class RockOutcrop:
    """Antarctic rock outcrop polygons (SCAR ADD) in longitude/latitude, indexed by bounding box.

    A polygon that crosses the antimeridian is made continuous (longitudes in 0..360) and stored twice,
    shifted by -360, so that the tiles on both sides of 180 deg see it.
    """

    def __init__(self, polygons: list[list[np.ndarray]]):
        self.shapes: list[dict] = []
        boxes = []
        for rings in polygons:                                        # one polygon = its rings (lon, lat) arrays
            lon = np.concatenate([r[:, 0] for r in rings])
            if lon.max() - lon.min() > 180:
                rings = [np.column_stack((np.where(r[:, 0] < 0, r[:, 0] + 360, r[:, 0]), r[:, 1])) for r in rings]
                shifts = (0.0, -360.0)
            else:
                shifts = (0.0,)
            for shift in shifts:
                moved = [r + np.array([shift, 0.0]) for r in rings]
                pts = np.concatenate(moved)
                boxes.append((*pts.min(axis=0), *pts.max(axis=0)))
                self.shapes.append({"type": "Polygon", "coordinates": moved})
        self.boxes = np.array(boxes, dtype=np.float64).reshape(-1, 4)   # lon_min, lat_min, lon_max, lat_max

    @classmethod
    def from_geometries(cls, geometries: list[dict]) -> RockOutcrop:
        """From GeoJSON-like (Multi)Polygons in EPSG:3031 metres (sources.read_gpkg_polygons)."""
        polygons = [rings for g in geometries
                    for rings in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"])]
        flat = [r for rings in polygons for r in rings]
        xy = np.concatenate(flat)
        lon, lat = polar_stereographic_south_to_lonlat(xy[:, 0], xy[:, 1])
        parts = np.split(np.column_stack((lon, lat)), np.cumsum([len(r) for r in flat])[:-1])
        out, i = [], 0
        for rings in polygons:
            out.append(parts[i:i + len(rings)])
            i += len(rings)
        return cls(out)

    def rasterize(self, lat0: int, lon0: int, shape: tuple[int, int]) -> np.ndarray:
        """(rows, cols) bool, True on rock, for the 1 x 1 deg cell (lat0, lon0), north-up."""
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        b = self.boxes
        hit = np.flatnonzero((b[:, 0] < lon0 + 1) & (b[:, 2] > lon0) & (b[:, 1] < lat0 + 1) & (b[:, 3] > lat0))
        if hit.size == 0:
            return np.zeros(shape, dtype=bool)
        shapes = [{"type": "Polygon", "coordinates": [r.tolist() for r in self.shapes[i]["coordinates"]]}
                  for i in hit]
        out = rasterize(shapes, out_shape=shape, fill=0, default_value=1, dtype="uint8",
                        transform=from_bounds(lon0, lat0, lon0 + 1, lat0 + 1, shape[1], shape[0]))
        return out.astype(bool)


_ROCK_LOCK = threading.Lock()
_ROCK: list[RockOutcrop] = []


def rock_outcrop() -> RockOutcrop:
    """The SCAR ADD rock outcrop layer, read once (thread-safe: tiles are processed in parallel)."""
    from src import sources

    with _ROCK_LOCK:
        if not _ROCK:
            _ROCK.append(RockOutcrop.from_geometries(sources.read_gpkg_polygons(sources.add_rock_outcrop_local())))
    return _ROCK[0]


def bare_fraction_from_classes(classes: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Bare share of each output pixel from WorldCover class samples; NaN when less than half of it is land
    (the rule of the Copernicus water mask for the topography)."""
    from src import sources

    land = sources.block_mean((~np.isin(classes, WORLDCOVER_NO_LAND)).astype(np.float32), out_shape)
    bare = sources.block_mean(np.isin(classes, WORLDCOVER_BARE).astype(np.float32), out_shape)
    return np.where(land >= 0.5, bare / np.maximum(land, 1e-9), np.nan)


def earth_tile_bare(name: str, lat0: int, lon0: int, worldcover: set[str]) -> tuple[np.ndarray, float]:
    """Bare-ground quantiles (1, L) over the ~463 m pixels of one Earth tile, and their land share.

    North of 60 S: ESA WorldCover classes, 25 samples per pixel. South of it: SCAR ADD rock outcrop
    rasterised on the same samples, on the land pixels of the Copernicus water mask (no vegetation there).
    """
    from src import sources

    shape = (data_prep.TOPO_PX_PER_DEG, data_prep.tile_columns(lat0))
    fine = (shape[0] * COVER_SUBSAMPLE, shape[1] * COVER_SUBSAMPLE)
    if lat0 < ANTARCTIC_LAT:
        dem, _ = data_prep.earth_tile_dem(name, lat0)
        frac = sources.block_mean(rock_outcrop().rasterize(lat0, lon0, fine).astype(np.float32), shape)
        frac[~np.isfinite(dem)] = np.nan
    elif sources.worldcover_file(lat0, lon0) in worldcover:
        frac = bare_fraction_from_classes(sources.read_worldcover(lat0, lon0, fine), shape)
    else:
        frac = np.full(shape, np.nan)
    return data_prep.sample_quantiles(frac)[None, :], float(np.isfinite(frac).mean())
