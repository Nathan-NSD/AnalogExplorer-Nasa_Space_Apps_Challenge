"""Tests of the remote-data readers that run offline (no network)."""
import numpy as np
import pytest

from src import sources

try:
    from compression import zstd  # standard library from Python 3.14 only, like the reader it tests
except ImportError:  # Python 3.12 of the CI (Streamlit Cloud): the app runs, the offline pipeline does not
    zstd = None


def _blosc_buffer(values: np.ndarray, blocksize: int, shuffle: str) -> bytes:
    """Minimal Blosc v1 encoder (zstd, no split) mirroring numcodecs, for round-trip tests."""
    raw = values.tobytes()
    typesize = values.dtype.itemsize
    flags = 0x10 | (4 << 5) | {"byte": 0x01, "bit": 0x04}[shuffle]
    blocks, payload = [], b""
    n_blocks = -(-len(raw) // blocksize)
    header_len = 16 + 4 * n_blocks
    for b in range(n_blocks):
        block = raw[b * blocksize:(b + 1) * blocksize]
        n = len(block) // typesize
        if shuffle == "byte":
            body = np.frombuffer(block, np.uint8, n * typesize).reshape(n, typesize).T.tobytes() + block[n * typesize:]
        else:  # bit shuffle (old v2 format: whole block only when n % 8 == 0)
            bits = np.unpackbits(np.frombuffer(block, np.uint8).reshape(n, typesize), axis=1, bitorder="little")
            body = np.packbits(bits.T, axis=1, bitorder="little").tobytes() if n % 8 == 0 else block
        comp = zstd.compress(body)
        blocks.append(header_len + len(payload))
        payload += len(comp).to_bytes(4, "little") + comp
    header = bytes([2, 1, flags, typesize]) + len(raw).to_bytes(4, "little") + blocksize.to_bytes(4, "little")
    header += (header_len + len(payload)).to_bytes(4, "little")
    return header + np.asarray(blocks, dtype="<i4").tobytes() + payload


@pytest.mark.skipif(zstd is None, reason="compression.zstd needs Python >= 3.14 (offline pipeline only)")
@pytest.mark.parametrize("shuffle", ["byte", "bit"])
def test_blosc_round_trip(shuffle):
    values = np.random.default_rng(0).normal(size=1000).astype("<f4")  # 2 blocks, the last one shorter
    assert np.array_equal(np.frombuffer(sources.blosc_decompress(_blosc_buffer(values, 2048, shuffle)), "<f4"),
                          values)


def test_aridity_index():
    pet = sources.hargreaves_pet(np.array([25.0]), np.array([35.0]), np.array([15.0]), np.array([400.0]))
    assert 4.0 < pet[0] < 9.0                                       # hot desert: several mm/day of demand
    arid = sources.aridity_from_climate(np.array([0.01, 10.0, 1.0, 1.0]), np.array([6.0, 4.0, 6.0, 0.0]),
                                        np.array([25.0, 20.0, -5.0, 10.0]))
    assert arid[0] > 0.99 and arid[1] == 0.0                        # hyperarid, humid
    assert np.isnan(arid[2]) and np.isnan(arid[3])                  # frozen month, no evaporative demand


def test_fixed_scale_offset_is_undone():
    stored = np.array([[100.0, -1000.0]], dtype=np.float32)       # (x - 1000) * 100
    out = sources._decode_filters(stored, [{"id": "fixedscaleoffset", "scale": 100, "offset": 1000}])
    assert np.allclose(out, [[1001.0, 990.0]])


# ------------------------------------------------ Mars Climate Database
def _mcd_file(lat_text: str = "18.38N", lon_text: str = "77.58E", ls: float = 90.0, rows: int = 25) -> str:
    header = ("##########\n### MCD_v6.2 with climatology average solar scenario.\n"
              f"### Ls {ls:.1f}deg. Latitude {lat_text}. Longitude {lon_text}. Altitude 10.0 m ALS.\n"
              "### Column 1 is Local time (Martian hour)\n### Column 2 is Surface temperature (K)\n##########\n")
    hours = np.arange(rows)
    temps = 235.0 + 40.0 * np.sin((hours - 7) / 24 * 2 * np.pi)
    return header + "".join(f"    {h:.5e}    {t:.5e}\n" for h, t in zip(hours, temps, strict=True))


def test_parse_mcd_table_skips_the_header():
    table = sources.parse_mcd_table(_mcd_file())
    assert table.shape == (25, 2) and table[0, 0] == 0 and table[-1, 0] == 24
    assert sources.mcd_header_position(_mcd_file("-5.37N", "-166.0E", 15.0)) == (-5.37, -166.0, 15.0)   # real format
    assert sources.mcd_header_position(_mcd_file("5.37S", "166.00W", 15.0)) == (-5.37, -166.0, 15.0)
    assert sources.mcd_header_position("no header") is None
    with pytest.raises(ValueError):
        sources.parse_mcd_table("### only a header\n")


def test_mcd_surface_temperature_reads_checks_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "mcdcgi.py" in url:
            return b"<html><a href='../txt/0a1b2c3d.txt'>Click here to download an ASCII file</a></html>"
        return _mcd_file().encode()

    monkeypatch.setattr(sources, "http_get", fake_get)
    cycle = sources.mcd_surface_temperature(18.38, 77.58, 90.0, cache_dir=tmp_path, pause_s=0)
    assert cycle.shape == (25,) and cycle.dtype == np.float32 and 190 < cycle.min() < cycle.max() < 280
    assert len(calls) == 2 and "latitude=18.38" in calls[0] and "ls=90.0" in calls[0] and "var1=tsurf" in calls[0]
    assert calls[1].endswith("/txt/0a1b2c3d.txt")
    again = sources.mcd_surface_temperature(18.38, 77.58, 90.0, cache_dir=tmp_path, pause_s=0)
    assert len(calls) == 2 and np.array_equal(again, cycle)            # second call served by the cache


def test_mcd_surface_temperature_rejects_wrong_answers(tmp_path, monkeypatch):
    def serve(text, page=b"<a href='../txt/ff00.txt'>file</a>"):
        monkeypatch.setattr(sources, "http_get", lambda url, **kw: page if "mcdcgi.py" in url else text.encode())

    serve(_mcd_file(), page=b"<html>error, no file</html>")
    with pytest.raises(RuntimeError):
        sources.mcd_surface_temperature(18.38, 77.58, 90.0, cache_dir=tmp_path, pause_s=0)
    serve(_mcd_file(rows=24))                                          # truncated cycle
    with pytest.raises(ValueError):
        sources.mcd_surface_temperature(18.38, 77.58, 90.0, cache_dir=tmp_path, pause_s=0)
    serve(_mcd_file(lon_text="77.58W"))                                # answer for another place
    with pytest.raises(ValueError):
        sources.mcd_surface_temperature(18.38, 77.58, 90.0, cache_dir=tmp_path, pause_s=0)
    serve(_mcd_file(ls=120.0))                                         # answer for another season
    with pytest.raises(ValueError):
        sources.mcd_surface_temperature(18.38, 77.58, 90.0, cache_dir=tmp_path, pause_s=0)
    assert not list(tmp_path.iterdir())                                # nothing wrong is ever cached
    serve(_mcd_file(lon_text="-166.0E", lat_text="46.0N"))             # west longitudes are understood
    assert sources.mcd_surface_temperature(46.0, -166.0, 90.0, cache_dir=tmp_path, pause_s=0).shape == (25,)


def test_worldcover_file_is_the_3_degree_cell_holding_the_tile():
    assert sources.worldcover_file(45, 6) == "ESA_WorldCover_10m_2021_v200_N45E006_Map.tif"
    assert sources.worldcover_file(47, 8) == "ESA_WorldCover_10m_2021_v200_N45E006_Map.tif"
    assert sources.worldcover_file(-25, -70) == "ESA_WorldCover_10m_2021_v200_S27W072_Map.tif"
    assert sources.worldcover_file(-1, -1) == "ESA_WorldCover_10m_2021_v200_S03W003_Map.tif"


def _gpkg_blob(rings: list) -> bytes:
    """GeoPackage geometry (little endian, xy envelope) holding one MultiPolygon made of one polygon."""
    import struct

    pts = np.concatenate(rings)
    envelope = struct.pack("<4d", pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
    wkb = struct.pack("<BII", 1, 6, 1) + struct.pack("<BII", 1, 3, len(rings))
    for ring in rings:
        wkb += struct.pack("<I", len(ring)) + np.asarray(ring, dtype="<f8").tobytes()
    return b"GP" + bytes([0, 0b011]) + struct.pack("<i", 3031) + envelope + wkb


def test_read_gpkg_polygons_decodes_the_geometry_blobs(tmp_path):
    import sqlite3

    outer = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]])
    hole = np.array([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 2.0]])
    path = tmp_path / "rock.gpkg"
    con = sqlite3.connect(path)
    con.execute("create table gpkg_geometry_columns (table_name text, column_name text)")
    con.execute("insert into gpkg_geometry_columns values ('rock', 'geom')")
    con.execute("create table rock (fid integer primary key, geom blob)")
    con.execute("insert into rock (geom) values (?)", (_gpkg_blob([outer, hole]),))
    con.commit()
    con.close()
    (geom,) = sources.read_gpkg_polygons(path)
    assert geom["type"] == "MultiPolygon"
    (rings,) = geom["coordinates"]
    assert np.array_equal(rings[0], outer) and np.array_equal(rings[1], hole)


def test_earth_minimum_temperature_is_the_mean_daily_minimum(monkeypatch):
    # The targets use the coldest hour of the mean cycle, so Earth uses TS_MIN_AVG, not the extreme TS_MIN
    def fake(store, variable):
        lat, lon = sources.POWER_STORES[store][1:]
        value = {"TS_MIN_AVG": -40.0, "TS_MIN": -60.0, "TS_AMP": 10.0}.get(variable, 5.0)
        return np.full((13, lat.size, lon.size), value, dtype=np.float32)

    monkeypatch.setattr(sources, "power_climatology", fake)
    fields = sources.earth_climate_fields()
    assert np.allclose(fields["temp_min_k"][0], -40.0 + 273.15)
    assert np.allclose(fields["diurnal_amp_k"][0], 20.0)          # TS_AMP is one half of the range
