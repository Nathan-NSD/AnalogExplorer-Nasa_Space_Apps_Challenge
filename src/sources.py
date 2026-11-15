"""Readers for the public remote datasets used to build the real dataset.

No Streamlit. Dependencies are kept minimal on purpose (Python 3.15 on Windows has no
wheel for numcodecs/zarr): HTTP with urllib, numpy, the standard-library zstd codec
(`compression.zstd`, Python >= 3.14: the offline pipeline needs it, the app does not) and
rasterio (imported lazily, only for rasters).

Sources
-------
- NASA POWER climatology (public AWS bucket, Zarr v2): Earth thermal, water, sunlight.
- Copernicus DEM GLO-90 (ESA, public AWS bucket, Cloud Optimized GeoTIFF): Earth topography.
- ESA WorldCover 10 m (public AWS bucket, COG) and SCAR ADD rock outcrop (BAS, GeoPackage): Earth bare ground.
- NASA PDS Geosciences (LRO LOLA, MGS MOLA, LRO Diviner): Moon and Mars targets.
- Mars Climate Database (LMD, with ESA and CNES): surface temperature of the Mars targets.
Downloads are cached under data/raw/ (ignored by Git).
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
USER_AGENT = "AnalogExplorer/1.0 (NASA Space Apps 2026 research prototype)"


# ===================================================================== HTTP
def http_get(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
    """GET a URL with a few retries (exponential back-off)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"download failed: {url}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def cached_download(url: str, path: Path) -> Path:
    """Download `url` to `path` once; later calls reuse the file."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(http_get(url, timeout=600))
        tmp.replace(path)
    return path


# ============================================================ Blosc (v1 format)
_BLOSC_ZSTD = 4


def _byte_unshuffle(block: bytes, typesize: int) -> bytes:
    """Inverse of Blosc's byte shuffle; trailing bytes that do not fill an element are kept."""
    n = len(block) // typesize
    head = np.frombuffer(block, dtype=np.uint8, count=n * typesize).reshape(typesize, n).T
    return head.tobytes() + block[n * typesize:]


def _bit_unshuffle(block: bytes, typesize: int, version: int) -> bytes:
    """Inverse of Blosc's bit shuffle (bitshuffle algorithm, little-endian bit order)."""
    n = len(block) // typesize
    if version <= 2:  # old format: only whole blocks with a multiple of 8 elements were shuffled
        if n % 8:
            return block
        elements = n
    else:
        elements = n - n % 8
    if elements == 0:
        return block
    nbytes = elements * typesize
    planes = np.frombuffer(block, dtype=np.uint8, count=nbytes).reshape(8 * typesize, elements // 8)
    bits = np.unpackbits(planes, axis=1, bitorder="little")                 # (bits per element, elements)
    out = np.packbits(bits.T, axis=1, bitorder="little")                     # (elements, typesize)
    return out.tobytes() + block[nbytes:]


def blosc_decompress(buffer: bytes) -> bytes:
    """Decompress a Blosc v1 buffer (as written by numcodecs.Blosc) compressed with zstd."""
    from compression import zstd  # standard library, Python >= 3.14

    version, flags, typesize = buffer[0], buffer[2], buffer[3]
    nbytes = int.from_bytes(buffer[4:8], "little")
    blocksize = int.from_bytes(buffer[8:12], "little")
    if flags & 0x02:  # stored without compression
        return bytes(buffer[16:16 + nbytes])
    codec = flags >> 5
    if codec != _BLOSC_ZSTD:
        raise NotImplementedError(f"Blosc codec {codec} not supported (only zstd)")
    byte_shuffle, bit_shuffle, no_split = bool(flags & 0x01), bool(flags & 0x04), bool(flags & 0x10)
    n_blocks = -(-nbytes // blocksize)
    starts = np.frombuffer(buffer, dtype="<i4", count=n_blocks, offset=16)
    out = bytearray()
    for b, start in enumerate(starts):
        bsize = min(blocksize, nbytes - b * blocksize)
        leftover = bsize < blocksize
        n_splits = typesize if (not no_split and not leftover) else 1
        sub_size = bsize // n_splits
        pos, block = int(start), bytearray()
        for _ in range(n_splits):
            cbytes = int.from_bytes(buffer[pos:pos + 4], "little")
            pos += 4
            chunk = buffer[pos:pos + cbytes]
            pos += cbytes
            block += chunk if cbytes == sub_size else zstd.decompress(chunk)
        block = bytes(block)
        if bit_shuffle:
            block = _bit_unshuffle(block, typesize, version)
        elif byte_shuffle:
            block = _byte_unshuffle(block, typesize)
        out += block
    if len(out) != nbytes:
        raise ValueError(f"Blosc: expected {nbytes} bytes, got {len(out)}")
    return bytes(out)


# ================================================================== Zarr v2
def _decode_filters(arr: np.ndarray, filters: list | None) -> np.ndarray:
    """Undo numcodecs filters (only FixedScaleOffset is used by POWER)."""
    for f in reversed(filters or []):
        if f["id"] != "fixedscaleoffset":
            raise NotImplementedError(f"Zarr filter {f['id']} not supported")
        arr = arr.astype(np.float32) / np.float32(f["scale"]) + np.float32(f["offset"])
    return arr


def read_zarr_array(store_url: str, name: str) -> np.ndarray:
    """Read a whole Zarr v2 array (Blosc/zstd chunks) over HTTP."""
    meta = json.loads(http_get(f"{store_url}/{name}/.zarray"))
    shape, chunks, sep = meta["shape"], meta["chunks"], meta.get("dimension_separator") or "."
    dtype = np.dtype(meta["dtype"])
    out = np.full(shape, np.nan if dtype.kind == "f" else 0, dtype=np.float32 if meta.get("filters") else dtype)
    grid = [range(-(-s // c)) for s, c in zip(shape, chunks, strict=True)]
    for index in np.ndindex(*[len(g) for g in grid]):
        key = sep.join(map(str, index))
        try:
            raw = blosc_decompress(http_get(f"{store_url}/{name}/{key}"))
        except RuntimeError:  # missing chunk = fill value
            continue
        block = np.frombuffer(raw, dtype=dtype).reshape(chunks)
        block = _decode_filters(block, meta.get("filters"))
        sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(index, chunks, shape, strict=True))
        out[sl] = block[tuple(slice(0, s.stop - s.start) for s in sl)]
    return out


# =============================================================== NASA POWER
POWER_BASE = "https://nasa-power.s3.amazonaws.com"
# store -> (path, latitudes, longitudes) of its regular grid (cell centres)
POWER_STORES = {
    "merra2": ("merra2/spatial/power_merra2_climatology_spatial_utc.zarr",
               np.linspace(-90.0, 90.0, 361), np.arange(576) * 0.625 - 180.0),
    "syn1deg": ("syn1deg/spatial/power_syn1deg_climatology_spatial_utc.zarr",
                np.arange(180) - 89.5, np.arange(360) - 179.5),
}


def _monthly_sun(lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per month and latitude: (daylight fraction, daily-mean top-of-atmosphere irradiance W/m2), (12, n_lat).

    Astronomical day length and extraterrestrial radiation on a horizontal surface
    (FAO-56 eq. 21 geometry), averaged over the days of each month.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    month_len = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    starts = np.concatenate([[0], np.cumsum(month_len)[:-1]])
    day, toa = np.empty((12, lat.size)), np.empty((12, lat.size))
    for m in range(12):
        doy = starts[m] + np.arange(month_len[m]) + 1
        decl = (np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0))[:, None]   # (days, 1)
        inv_r2 = (1 + 0.033 * np.cos(2 * np.pi * doy / 365.0))[:, None]                # Earth-Sun distance
        h0 = np.arccos(np.clip(-np.tan(lat)[None, :] * np.tan(decl), -1.0, 1.0))        # (days, n_lat)
        day[m] = (h0 / np.pi).mean(axis=0)
        toa[m] = (1361.0 / np.pi * inv_r2 * (h0 * np.sin(lat) * np.sin(decl)
                                             + np.cos(lat) * np.cos(decl) * np.sin(h0))).mean(axis=0)
    return day, toa


def daylight_fraction(lat_deg: np.ndarray) -> np.ndarray:
    """Mean fraction of the day with the Sun above the horizon, per month: (12, n_lat).

    Polar night gives 0 and polar day gives 1 (no refraction).
    """
    return _monthly_sun(lat_deg)[0]


def hargreaves_pet(t_mean: np.ndarray, t_max: np.ndarray, t_min: np.ndarray, toa_wm2: np.ndarray) -> np.ndarray:
    """Potential evapotranspiration (mm/day), Hargreaves & Samani 1985 (FAO-56 eq. 52).

    Temperatures in C (daily mean, mean daily max, mean daily min); toa_wm2: extraterrestrial
    radiation. PET = 0.0023 Ra (T + 17.8) sqrt(Tmax - Tmin), Ra in mm/day of evaporation.
    """
    ra_mm = np.asarray(toa_wm2) * 0.0864 * 0.408                          # W/m2 -> MJ/m2/day -> mm/day
    pet = 0.0023 * ra_mm * (np.asarray(t_mean) + 17.8) * np.sqrt(np.clip(np.asarray(t_max) - t_min, 0.0, None))
    return np.clip(pet, 0.0, None)


def aridity_from_climate(precip_mm_day: np.ndarray, pet_mm_day: np.ndarray, t_mean_c: np.ndarray) -> np.ndarray:
    """Monthly aridity in [0, 1]: 1 - min(P / PET, 1), the UNEP aridity index (P/PET) turned into a deficit.

    1 = hyperarid (P << PET), 0 = humid (P >= PET). Months below 0 C (no liquid water, covered by
    the frozen-ground variable) and months without evaporative demand are NaN.
    """
    p, pet = np.asarray(precip_mm_day, dtype=np.float64), np.asarray(pet_mm_day, dtype=np.float64)
    ok = (np.asarray(t_mean_c) >= 0.0) & (pet > 0.1) & np.isfinite(p)
    ratio = np.divide(p, pet, out=np.full(p.shape, np.nan), where=ok)
    return np.where(ok, 1.0 - np.minimum(ratio, 1.0), np.nan).astype(np.float32)


def earth_climate_fields() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Monthly Earth fields for the six climate variables of config.VARIABLES.

    Returns {variable name: (field (12, n_lat, n_lon), latitudes, longitudes)}, in the
    units of config (K, 0-1, W/m2). Physical definitions:
    - diurnal_amp_k: mean daily max - mean daily min of skin temperature (2 x TS_AMP; POWER defines
      TS_AMP as one half of the skin temperature range);
    - temp_min_k: mean daily minimum of skin temperature (TS_MIN_AVG), the counterpart of the coldest
      hour of the mean diurnal cycle used for the Moon (Diviner) and Mars (MCD); TS_MIN, the coldest
      hourly value of the month, is an extreme 2 K colder in median and 5-8 K in cold deserts;
    - aridity: 1 - min(P / PET, 1) per month above 0 C, P = corrected precipitation (PRECTOTCORR),
      PET = Hargreaves from air temperature (T2M, T2M_MAX_AVG, T2M_MIN_AVG) and astronomical
      top-of-atmosphere radiation (UNEP aridity index; replaces the root-zone soil wetness of
      the first version, which stayed near 0.67 in the hyperarid Atacama);
    - ice_proxy: 1 when the ground is frozen at depth (soil layer 6, 3-10 m, below 0 C;
      on land ice, where soil is not modelled, the skin temperature is used);
    - insolation_wm2: all-sky shortwave irradiance on a horizontal surface;
    - sunlit_frac: astronomical daylight fraction x (1 - daytime cloud fraction).
    """
    m_lat, m_lon = POWER_STORES["merra2"][1:]
    s_lat, s_lon = POWER_STORES["syn1deg"][1:]
    merra = {v: power_climatology("merra2", v)[:12] for v in ("TS_AMP", "TS_MIN_AVG", "TSOIL6", "TS", "T2M",
                                                              "T2M_MAX_AVG", "T2M_MIN_AVG", "PRECTOTCORR")}
    syn = {v: power_climatology("syn1deg", v)[:12] for v in ("ALLSKY_SFC_SW_DWN", "CLOUD_AMT_DAY")}

    frozen = np.where(np.isfinite(merra["TSOIL6"]), merra["TSOIL6"] < 0.0, merra["TS"] < 0.0).astype(np.float32)
    toa = _monthly_sun(m_lat)[1][:, :, None]                                      # (12, n_lat, 1)
    pet = hargreaves_pet(merra["T2M"], merra["T2M_MAX_AVG"], merra["T2M_MIN_AVG"], toa)
    aridity = aridity_from_climate(merra["PRECTOTCORR"], pet, merra["T2M"])
    day = daylight_fraction(s_lat)[:, :, None]                                  # (12, n_lat, 1)
    clear = 1.0 - np.clip(syn["CLOUD_AMT_DAY"], 0.0, 100.0) / 100.0
    sunlit = np.where(day > 0, day * np.where(np.isfinite(clear), clear, 1.0), 0.0)
    return {
        "diurnal_amp_k": (2.0 * merra["TS_AMP"], m_lat, m_lon),
        "temp_min_k": (merra["TS_MIN_AVG"] + 273.15, m_lat, m_lon),
        "aridity": (aridity, m_lat, m_lon),
        "ice_proxy": (frozen, m_lat, m_lon),
        "insolation_wm2": (syn["ALLSKY_SFC_SW_DWN"], s_lat, s_lon),
        "sunlit_frac": (sunlit.astype(np.float32), s_lat, s_lon),
    }


# ======================================================== Copernicus DEM GLO-90
COPERNICUS_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
GDAL_REMOTE_OPTIONS = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
                       "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.img,.xml,.lbl",
                       "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "1", "VSI_CACHE": "TRUE"}


def copernicus_tiles() -> list[tuple[str, int, int]]:
    """All GLO-90 land tiles as (name, south latitude, west longitude); 1 deg x 1 deg each."""
    path = cached_download(f"{COPERNICUS_BASE}/tileList.txt", RAW_DIR / "copernicus" / "tileList.txt")
    tiles = []
    for name in path.read_text().split():
        parts = name.split("_")                       # Copernicus_DSM_COG_30_N45_00_E006_00_DEM
        lat = int(parts[4][1:]) * (1 if parts[4][0] == "N" else -1)
        lon = int(parts[6][1:]) * (1 if parts[6][0] == "E" else -1)
        tiles.append((name, lat, lon))
    return tiles


def block_mean(a: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Average of a 2-D array over the blocks of a coarser grid (output no larger than the input)."""
    rows = np.linspace(0, a.shape[0], out_shape[0] + 1).astype(int)
    cols = np.linspace(0, a.shape[1], out_shape[1] + 1).astype(int)
    sums = np.add.reduceat(np.add.reduceat(a, rows[:-1], axis=0), cols[:-1], axis=1)
    return sums / (np.diff(rows)[:, None] * np.diff(cols)[None, :])


def read_copernicus_tile(name: str, out_shape: tuple[int, int]) -> np.ndarray:
    """Elevation (m) of one GLO-90 tile resampled to `out_shape` (average), north-up, water masked (NaN).

    The DEM is read remotely from the internal overviews of the COG (a few hundred kB).
    Water comes from the tile's Water Body Mask (AUXFILES/<tile>_WBM.tif, same grid, ~19 kB;
    0 land, 1 ocean, 2 lake, 3 river), read at full resolution: an output pixel is masked
    when at least half of it is water. Without the mask, falls back to 'elevation == 0' (ocean).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.errors import RasterioIOError

    base = f"/vsicurl/{COPERNICUS_BASE}/{name}"
    with rasterio.Env(**GDAL_REMOTE_OPTIONS):
        with rasterio.open(f"{base}/{name}.tif") as src:
            z = src.read(1, out_shape=out_shape, resampling=Resampling.average).astype(np.float32)
        try:
            with rasterio.open(f"{base}/AUXFILES/{name.removesuffix('_DEM')}_WBM.tif") as wbm:
                water = block_mean((wbm.read(1) > 0).astype(np.float32), out_shape)
            z[water >= 0.5] = np.nan
        except RasterioIOError:
            z[z == 0.0] = np.nan
    return z


# ===================================================== Earth surface cover
WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
WORLDCOVER_CITATION = "ESA WorldCover 10 m 2021 v200 (Zanaga et al. 2022), CC BY 4.0"
ADD_ROCK_URL = ("https://ramadda.data.bas.ac.uk/repository/entry/get/add_rock_outcrop_high_res_polygon_v7_11.gpkg"
                "?entryid=synth:815525ca-cefe-4fbf-9224-3ee784e7de4e:"
                "L2FkZF9yb2NrX291dGNyb3BfaGlnaF9yZXNfcG9seWdvbl92N18xMS5ncGtn")
ADD_CITATION = ("SCAR Antarctic Digital Database, high-resolution rock outcrop v7.11 (Sentinel-2, 2024/25; "
                "doi:10.5285/815525ca-cefe-4fbf-9224-3ee784e7de4e), CC BY 4.0")


def worldcover_file(lat0: int, lon0: int) -> str:
    """Name of the 3 x 3 deg WorldCover file holding the 1 x 1 deg cell whose south-west corner is (lat0, lon0)."""
    lat3, lon3 = 3 * (lat0 // 3), 3 * (lon0 // 3)
    return (f"ESA_WorldCover_10m_2021_v200_{'N' if lat3 >= 0 else 'S'}{abs(lat3):02d}"
            f"{'E' if lon3 >= 0 else 'W'}{abs(lon3):03d}_Map.tif")


def worldcover_files() -> set[str]:
    """Every WorldCover file that exists (official grid: land between 60 S and 84 N)."""
    path = cached_download(f"{WORLDCOVER_BASE}/esa_worldcover_grid.geojson",
                           RAW_DIR / "landcover" / "esa_worldcover_grid.geojson")
    grid = json.loads(path.read_text(encoding="utf-8"))
    return {f"ESA_WorldCover_10m_2021_v200_{f['properties']['ll_tile']}_Map.tif" for f in grid["features"]}


def read_worldcover(lat0: int, lon0: int, out_shape: tuple[int, int]) -> np.ndarray:
    """WorldCover classes of one 1 x 1 deg cell sampled on `out_shape` (nearest, north-up).

    Read remotely from the internal overviews of the COG; the caller checks that the file exists
    (worldcover_files), so a read error is a real failure, retried on the next run.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    with rasterio.Env(**GDAL_REMOTE_OPTIONS), \
            rasterio.open(f"/vsicurl/{WORLDCOVER_BASE}/v200/2021/map/{worldcover_file(lat0, lon0)}") as src:
        window = from_bounds(lon0, lat0, lon0 + 1, lat0 + 1, transform=src.transform)
        return src.read(1, window=window, out_shape=out_shape, resampling=Resampling.nearest)


def add_rock_outcrop_local() -> Path:
    """SCAR ADD high-resolution rock outcrop polygons of Antarctica (GeoPackage, EPSG:3031, ~290 MB, cached)."""
    return cached_download(ADD_ROCK_URL, RAW_DIR / "landcover" / "add_rock_outcrop_high_res_polygon_v7_11.gpkg")


_GPKG_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def wkb_to_geojson(buf: bytes, pos: int = 0) -> tuple[dict, int]:
    """Decode one ISO WKB Polygon or MultiPolygon into a GeoJSON-like dict; returns (geometry, end position).

    Each ring is an (n, 2) float64 array (millions of vertices fit in memory, unlike Python lists).
    Z and M values are dropped. Only the polygon types used by the ADD layers are supported.
    """
    import struct

    order = "<" if buf[pos] == 1 else ">"
    code = struct.unpack_from(order + "I", buf, pos + 1)[0]
    kind, dims = code % 1000, (2, 3, 3, 4)[code // 1000]
    pos += 5
    if kind == 6:                                                   # MultiPolygon: a list of Polygon WKB
        (n,) = struct.unpack_from(order + "I", buf, pos)
        pos += 4
        polygons = []
        for _ in range(n):
            part, pos = wkb_to_geojson(buf, pos)
            polygons.append(part["coordinates"])
        return {"type": "MultiPolygon", "coordinates": polygons}, pos
    if kind != 3:
        raise ValueError(f"unsupported WKB geometry type {code}")
    (n_rings,) = struct.unpack_from(order + "I", buf, pos)
    pos += 4
    rings = []
    for _ in range(n_rings):
        (n_pts,) = struct.unpack_from(order + "I", buf, pos)
        pos += 4
        pts = np.frombuffer(buf, dtype=order + "f8", count=n_pts * dims, offset=pos).reshape(n_pts, dims)
        rings.append(np.array(pts[:, :2]))
        pos += 8 * n_pts * dims
    return {"type": "Polygon", "coordinates": rings}, pos


def read_gpkg_polygons(path: Path, table: str | None = None) -> list[dict]:
    """GeoJSON-like (Multi)Polygons of a GeoPackage feature table, in the layer's own CRS (sqlite3 only).

    GeoPackage geometry blob: 'GP', version, flags (bits 1-3: envelope kind), srs id, envelope, then WKB.
    """
    import sqlite3

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if table is None:
            table, column = con.execute("select table_name, column_name from gpkg_geometry_columns").fetchone()
        else:
            column = con.execute("select column_name from gpkg_geometry_columns where table_name = ?",
                                 (table,)).fetchone()[0]
        out = []
        for (blob,) in con.execute(f'select "{column}" from "{table}" where "{column}" is not null'):
            if blob[:2] != b"GP":
                raise ValueError("not a GeoPackage geometry blob")
            start = 8 + _GPKG_ENVELOPE_BYTES[(blob[3] >> 1) & 0b111]
            out.append(wkb_to_geojson(blob, start)[0])
        return out
    finally:
        con.close()


# ===================================================== NASA PDS (Moon and Mars)
LOLA_BASE = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr"
LOLA_POLAR_SOUTH = f"{LOLA_BASE}/polar/float_img/ldem_75s_240m_float"        # 240 m, km above 1737.4 km
LOLA_GLOBAL = f"{LOLA_BASE}/cylindrical/float_img/ldem_64_float"             # 1/64 deg ~ 474 m, km
MOLA_BASE = "https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/meg128"
DIVINER_GCP = ("https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gcp/"
               "global_cumul_avg_cyl_{band}_002.tab")


def mola_tile(lat: float, lon_east: float) -> str:
    """URL (PDS4 label) of the MOLA MEGDR 128 ppd tile containing a point (lon in 0-360 E)."""
    top = next(t for t in (88, 44, 0, -44) if lat <= t and lat > t - 44)
    left = int(lon_east // 90) * 90
    return f"{MOLA_BASE}/megt{abs(top):02d}{'n' if top >= 0 else 's'}{left:03d}hb.xml"


PGDA_ILLUMINATION_SOUTH = "https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_75S_120M_201608.TIF"
# NASA PGDA average solar illumination (Mazarico et al. 2011), 75-90 S, 120 m, int16 x file scale = fraction of time lit


def read_dem_window(source: str, lat: float, lon: float, half_km: float, radius_km: float,
                    pixel_m: float, projection: str, unit_m: float = 1.0,
                    scale_from_file: bool = False) -> tuple[np.ndarray, tuple]:
    """Square DEM window of side 2 x half_km on the ground, resampled to ~pixel_m (average).

    source: local path or URL of a GDAL-readable raster (PDS4 label). projection:
    "polar" (stereographic, true scale near the pole) or "equirectangular" (true scale
    at the equator: the window is widened by 1/cos(lat) and dx shrinks with latitude).
    Returns (elevation in metres, (dy, dx per row) in metres on the ground).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform
    from rasterio.windows import from_bounds

    path = source if Path(source).exists() else f"/vsicurl/{source}"
    with rasterio.Env(**GDAL_REMOTE_OPTIONS), rasterio.open(path) as src:
        geo = f"+proj=longlat +R={radius_km * 1000:.0f} +no_defs"
        (x,), (y,) = transform(geo, src.crs, [lon], [lat])
        widen = 1.0 / max(np.cos(np.radians(lat)), 1e-3) if projection == "equirectangular" else 1.0
        hx, hy = half_km * 1000 * widen, half_km * 1000
        n = max(8, int(round(2 * half_km * 1000 / pixel_m)))
        win = from_bounds(x - hx, y - hy, x + hx, y + hy, transform=src.transform)
        z = src.read(1, window=win, out_shape=(n, n), resampling=Resampling.average,
                     masked=True).astype(np.float32).filled(np.nan) * unit_m
        if scale_from_file:  # packed products (e.g. int16 x scale + offset)
            z = z * np.float32(src.scales[0]) + np.float32(src.offsets[0])
    dy = 2 * hy / n
    if projection == "equirectangular":
        rows_lat = lat + np.degrees(hy / (radius_km * 1000)) * (1 - (2 * np.arange(n) + 1) / n)
        dx = 2 * hx / n * np.cos(np.radians(rows_lat))
    else:
        dx = np.full(n, 2 * hx / n)
    return z, (dy, dx)


def lola_polar_local() -> Path:
    """Download the LOLA south polar 240 m DEM (58 MB) once; returns the local PDS4 label path."""
    for ext in (".img", ".xml"):
        cached_download(LOLA_POLAR_SOUTH + ext, RAW_DIR / "lola" / (Path(LOLA_POLAR_SOUTH).name + ext))
    return RAW_DIR / "lola" / (Path(LOLA_POLAR_SOUTH).name + ".xml")


def diviner_cells(band: str) -> np.ndarray:
    """Diviner Global Cumulative Product of one 10-deg band: (n, 4) lon, lat, local time, Tbol (K).

    Bins of 0.5 deg x 0.5 deg x 0.25 h, averaged over 2009-2015; -9999 (no data) -> NaN.
    The ~156 MB table is downloaded once to data/raw/diviner/.
    """
    import pandas as pd

    url = DIVINER_GCP.format(band=band)
    parsed = RAW_DIR / "diviner" / f"gcp_{band}.npy"
    if parsed.exists():
        return np.load(parsed)
    path = cached_download(url, RAW_DIR / "diviner" / Path(url).name)
    df = pd.read_csv(path, header=None, usecols=[0, 1, 2, 10], skipinitialspace=True, dtype=str)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=[0, 1, 2])   # drops any text line
    arr = np.array(df.to_numpy(dtype=np.float64), copy=True)
    arr[arr[:, 3] < 0, 3] = np.nan
    np.save(parsed, arr.astype(np.float32))
    return arr


# ============================================ Mars Climate Database (web interface)
MCD_BASE = "https://www-mars.lmd.jussieu.fr/mcd_python"
MCD_CITATION = ("Mars Climate Database v6.2 (c) LMD/OU/IAA/ESA/CNES, web interface, climatology scenario with "
                "average solar EUV, high-resolution mode")


def parse_mcd_table(text: str) -> np.ndarray:
    """Numeric table (rows, columns) of an ASCII file of the MCD web interface ('#' lines are its header)."""
    rows = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not rows:
        raise ValueError("MCD answer holds no data row")
    return np.array(rows, dtype=np.float64)


def mcd_header_position(text: str) -> tuple[float, float, float] | None:
    """(latitude, east longitude, Ls) in degrees echoed by the header of an MCD ASCII file; None if absent.

    The interface writes signed values with a fixed letter ('Latitude -5.37N. Longitude -166.0E.').
    """
    place = re.search(r"Latitude\s+(-?[0-9.]+)([NS])\.\s+Longitude\s+(-?[0-9.]+)([EW])", text)
    season = re.search(r"Ls\s+([0-9.]+)deg", text)
    if place is None or season is None:
        return None
    lat = float(place.group(1)) * (1 if place.group(2) == "N" else -1)
    lon = float(place.group(3)) * (1 if place.group(4) == "E" else -1)
    return lat, lon, float(season.group(1))


def mcd_surface_temperature(lat: float, lon: float, ls: float, cache_dir: Path | None = None,
                            pause_s: float = 1.0) -> np.ndarray:
    """Mean diurnal cycle of the surface temperature (K) at one point of Mars and one solar longitude.

    25 values, one per Martian hour of local time from 0 to 24, read from the Mars Climate
    Database web interface (see MCD_CITATION). lon in degrees east. One request per call,
    followed by a pause: the interface is meant for moderate use. Cached in data/raw/mcd/.
    """
    cache = (cache_dir or RAW_DIR / "mcd") / f"tsurf_{lat:+07.2f}_{lon:+08.2f}_ls{ls:05.1f}.npy"
    if cache.exists():
        return np.load(cache)
    query = (f"var1=tsurf&var2=none&var3=none&var4=none&datekeyhtml=1&ls={ls:.1f}&localtime=all"
             f"&latitude={lat:.2f}&longitude={lon:.2f}&altitude=10.&zkey=3&isfixedlt=off&dust=1&hrkey=1"
             "&averaging=off&dpi=80&islog=off&colorm=jet&proj=cyl")
    page = http_get(f"{MCD_BASE}/cgi-bin/mcdcgi.py?{query}", timeout=180).decode("utf-8", "replace")
    link = re.search(r"txt/([0-9a-f]+\.txt)", page)
    if link is None:
        raise RuntimeError(f"MCD web interface returned no data file for lat={lat}, lon={lon}, Ls={ls}")
    text = http_get(f"{MCD_BASE}/txt/{link.group(1)}").decode("utf-8", "replace")
    table = parse_mcd_table(text)
    where = f"lat={lat}, lon={lon}, Ls={ls}"
    if table.shape != (25, 2) or not np.allclose(table[:, 0], np.arange(25)):
        raise ValueError(f"unexpected MCD table for {where}: shape {table.shape}")
    tsurf = table[:, 1]
    if not ((tsurf > 100) & (tsurf < 350)).all():
        raise ValueError(f"MCD surface temperature out of the physical range for {where}")
    echoed = mcd_header_position(text)      # the answer must be for the point and season that were asked
    if echoed is None or abs(echoed[0] - lat) > 0.02 or abs((echoed[1] - lon + 180) % 360 - 180) > 0.02 \
            or abs(echoed[2] - ls) > 0.06:
        raise ValueError(f"MCD answered for {echoed}, not for {where}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, tsurf.astype(np.float32))
    time.sleep(pause_s)
    return tsurf.astype(np.float32)


def power_climatology(store: str, variable: str, cache_dir: Path | None = None) -> np.ndarray:
    """Monthly climatology of a POWER variable: (13, n_lat, n_lon), months 1-12 then the annual value.

    Cached as .npy under data/raw/power/ after the first download.
    """
    path, lat, lon = POWER_STORES[store]
    cache = (cache_dir or RAW_DIR / "power") / f"{store}_{variable}.npy"
    if cache.exists():
        return np.load(cache)
    arr = read_zarr_array(f"{POWER_BASE}/{path}", variable)
    if arr.shape[1:] != (lat.size, lon.size):
        raise ValueError(f"unexpected POWER grid {arr.shape} for {store}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr
