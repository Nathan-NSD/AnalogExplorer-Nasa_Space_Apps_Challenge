"""Tests of the data preparation blocks and of the data contract (synthetic samples, no network)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import config, data_prep
from tests import synthetic


@pytest.fixture(scope="module")
def small_ds():
    return synthetic.build_dataset(n_clusters=20, tiles_per_cluster=10, seed=1)


def test_slope_of_a_plane():
    x = np.arange(10, dtype=np.float32)
    dem = np.tile(x, (10, 1))  # rises 1 m per 1 m pixel -> 45 degrees
    assert np.allclose(data_prep.slope_degrees(dem, 1.0), 45.0)


def test_slope_on_a_geographic_grid():
    lat = np.linspace(60.0, 61.0, 50)                 # high latitude: 1 deg of longitude ~ half a degree of latitude
    dy, dx = data_prep.geographic_pixel_size(lat, 0.02, 0.02, 6371.0)
    x_m = np.cumsum(np.tile(dx[:, None], (1, 40)), axis=1)
    dem = 0.1 * x_m                                    # rises 10 cm per metre eastward everywhere
    slope = data_prep.slope_degrees(dem, (dy, dx))
    assert np.allclose(slope, np.degrees(np.arctan(0.1)), atol=1e-3)
    assert not np.allclose(data_prep.slope_degrees(dem, dy), np.degrees(np.arctan(0.1)), atol=1.0)


def test_weighted_quantiles():
    q = data_prep.weighted_quantiles([0.0, 10.0], [1.0, 3.0], np.array([0.5]))
    assert 5.0 < q[0] <= 10.0                          # the heavier value pulls the median up
    assert np.allclose(data_prep.weighted_quantiles([1, 2, 3], [1, 1, 1], np.array([0.5])), 2.0)


def test_orbital_illumination_of_the_lunar_equator():
    insol, day, _ = data_prep.orbital_illumination(0.0, "Moon")
    assert np.allclose(day, 0.5, atol=1e-3)
    assert np.allclose(insol, 1361.0 / np.pi, rtol=0.04)   # S0/pi on a flat equatorial site
    insol_mars, _, _ = data_prep.orbital_illumination(0.0, "Mars")
    assert 120 < insol_mars.mean() < 220                    # weaker sunlight at 1.52 AU


def test_earth_climate_quantiles_with_fallback():
    lat = np.array([10.25, 10.75, 11.25])
    lon = np.array([20.25, 20.75])
    field = np.arange(12 * 3 * 2, dtype=np.float32).reshape(12, 3, 2)
    fields = {name: (field, lat, lon) for name in data_prep.CLIMATE_VARIABLES}
    fields["aridity"] = (np.full((12, 3, 2), np.nan, np.float32), lat, lon)   # ice sheet: no soil model
    q = data_prep.earth_climate_quantiles(fields, 10, 20)
    i = data_prep.CLIMATE_VARIABLES.index("aridity")
    assert np.allclose(q[i], 1.0)
    j = data_prep.CLIMATE_VARIABLES.index("diurnal_amp_k")
    assert q[j, 0] >= field[:, :2].min() and q[j, -1] <= field[:, :2].max()   # rows 10.25 and 10.75 only


def test_target_variable_may_be_not_evaluated(small_ds):
    tq = small_ds.target_quantiles.copy()
    tq[0, 2] = np.nan                                  # whole variable missing: allowed
    ds = dataclasses.replace(small_ds, target_quantiles=tq)
    ds.validate()
    tq[0, 3, :5] = np.nan                              # partially missing: rejected
    with pytest.raises(ValueError):
        ds.validate()


def test_synthetic_dataset_follows_the_contract(small_ds):
    small_ds.validate()
    assert small_ds.quantiles.shape[1] == len(config.VARIABLES)
    assert small_ds.meta["source"] == "synthetic"


def test_no_silent_fallback_without_a_dataset(monkeypatch):
    """Without a dataset on disk the app and the atlas stop, they never invent one."""
    monkeypatch.setattr(data_prep, "find_dataset_dir", lambda: None)
    with pytest.raises(FileNotFoundError, match="--real"):
        data_prep.load_default_dataset()
    with pytest.raises(SystemExit):
        data_prep.main(["--mock"])                        # the synthetic dataset is no longer an option


def test_default_dataset_is_real_v1_unless_another_is_named(tmp_path, monkeypatch):
    """A sample (real_sample sorts before real_v1) is loaded only when named; a missing named dataset is not
    replaced by another one."""
    for name in ("real_sample", "real_v1"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(data_prep, "DEFAULT_DATASET_DIR", tmp_path / "real_v1")
    monkeypatch.delenv(data_prep.DATASET_ENV_VAR, raising=False)
    assert data_prep.find_dataset_dir() == tmp_path / "real_v1"
    monkeypatch.setenv(data_prep.DATASET_ENV_VAR, str(tmp_path / "real_sample"))
    assert data_prep.find_dataset_dir() == tmp_path / "real_sample"
    monkeypatch.setenv(data_prep.DATASET_ENV_VAR, str(tmp_path / "missing"))
    assert data_prep.find_dataset_dir() is None
    with pytest.raises(FileNotFoundError, match="missing"):
        data_prep.load_default_dataset()


def test_save_load_roundtrip(small_ds, tmp_path):
    data_prep.save_dataset(small_ds, tmp_path)
    back = data_prep.load_dataset(tmp_path)
    assert np.array_equal(back.quantiles, small_ds.quantiles)
    assert list(back.targets["target_id"]) == list(small_ds.targets["target_id"])


def test_sourced_catalogue():
    known = data_prep.known_analogs_table()
    assert known["source"].str.len().min() > 20                     # every site carries its sources
    mars = data_prep.analogs_relevant_to(known, "Mars")
    moon = data_prep.analog_properties(known, "Moon")
    assert mars.sum() >= 5 and sum(bool(p) for p in moon) >= 5     # enough sites for a conclusive test
    for body in ("Mars", "Moon"):                                   # every coded property exists
        for props in data_prep.analog_properties(known, body):
            data_prep.property_weights(props)


def test_property_weights_test_the_exact_property():
    names = config.VARIABLE_NAMES
    w = data_prep.property_weights(("aridity",))                    # a desert famous for aridity only
    assert w[names.index("aridity")] == 1.0 and w.sum() == 1.0      # frozen ground plays no part
    w = data_prep.property_weights(("aridity", "Topography"))       # each property weighs the same
    assert w[names.index("aridity")] == 0.5 and w[names.index("slope_deg")] == 0.25
    usable = np.ones(len(names), dtype=bool)
    usable[names.index("temp_min_k")] = False                       # out of Earth's range for the target
    assert data_prep.property_weights(("temp_min_k",), usable).sum() == 0
    with pytest.raises(ValueError):
        data_prep.property_weights(("Water / Ice ",))


def test_analogs_relevant_to():
    df = pd.DataFrame({"name": ["a", "b", "c"], "relevant_to": ["Mars", "Moon;Mars", None]})
    assert data_prep.analogs_relevant_to(df, "Moon").tolist() == [False, True, False]
    assert data_prep.analogs_relevant_to(df.drop(columns="relevant_to"), "Moon").all()


# ------------------------------------------------------------ fine-scale zoom
def test_block_edges_cover_the_raster():
    assert data_prep.block_edges(240, 24).tolist() == list(range(0, 241, 24))
    assert data_prep.block_edges(40, 24).tolist() == [0, 20, 40]          # 2 blocks of 20 px rather than 24 + 16
    assert data_prep.block_edges(10, 24).tolist() == [0, 10]              # never less than one block


def test_subarea_topography_separates_flat_and_rough_ground():
    rng = np.random.default_rng(0)
    dem = np.zeros((48, 72), dtype=np.float32)
    dem[:, 48:] = rng.normal(0.0, 200.0, (48, 24))       # rough eastern third
    dem[:24, :24] = np.nan                               # water in the north-west block
    q, land, grid = data_prep.subarea_topography(dem, 463.0)
    assert grid == (2, 3) and q.shape == (6, 2, config.QUANTILE_LEVELS.size)
    assert np.isnan(q[0]).all() and land[0] == 0.0       # not enough land: no statistics
    median = config.QUANTILE_LEVELS.size // 2
    assert q[4, 1, median] == pytest.approx(0.0, abs=1e-6)                # flat block: no roughness
    assert q[5, 1, median] > 50.0 and q[5, 0, median] > q[4, 0, median]   # rough block
    assert (np.diff(q[5], axis=-1) >= 0).all()


def test_tile_zoom_bounds_tile_the_cell():
    dem = np.zeros((240, 120), dtype=np.float32)
    q, land, grid = data_prep.subarea_topography(dem, 463.0)
    tile = data_prep.TileZoom(60, -10, dem, q, land, grid)
    bounds = tile.bounds()
    assert bounds.shape == (grid[0] * grid[1], 4)
    assert bounds[:, 0].min() == 60 and bounds[:, 1].max() == 61
    assert bounds[:, 2].min() == -10 and bounds[:, 3].max() == -9
    assert bounds[0, 1] == 61 and bounds[0, 2] == -10                    # first sub-area: north-west corner
    dy, dx = tile.pixel_size_m()
    assert dy == pytest.approx(463.3, abs=0.5) and dx.shape == (240,) and dx.mean() == pytest.approx(463.0, rel=0.03)
    assert data_prep.zoom_key(60.5, -9.5) == (60, -10) and data_prep.zoom_key(-0.5, 0.5) == (-1, 0)


def test_zoom_round_trip_with_the_dataset(tmp_path):
    ds = synthetic.build_dataset(n_clusters=20, tiles_per_cluster=10, seed=1, with_zoom=True)
    assert ds.zoom and set(ds.target_dems) == set(ds.targets["target_id"])
    back = data_prep.load_dataset(data_prep.save_dataset(ds, tmp_path / "with_zoom"))
    assert set(back.zoom) == set(ds.zoom)
    key = next(iter(ds.zoom))
    a, b = ds.zoom[key], back.zoom[key]
    assert b.grid == a.grid and np.allclose(b.quantiles, a.quantiles, equal_nan=True)
    assert np.allclose(b.dem, a.dem, atol=0.5)            # elevation stored to the metre
    packed = data_prep._pack_dem(np.array([[1.4, np.nan]]))
    assert np.isnan(data_prep._unpack_dem(packed)[0, 1]) and data_prep._unpack_dem(packed)[0, 0] == 1.0
    plain = data_prep.load_dataset(data_prep.save_dataset(
        synthetic.build_dataset(n_clusters=20, tiles_per_cluster=10, seed=1), tmp_path / "plain"))
    assert plain.zoom == {} and plain.target_dems == {}   # the zoom file is optional


# -------------------------------------------------------- subsurface case studies
def test_lava_tube_catalogue_is_well_formed():
    names = [s[0] for s in config.LAVA_TUBE_SITES]
    assert len(set(names)) == len(names) >= 15
    assert all(-90 <= lat <= 90 and -180 <= lon <= 180 for _, _, lat, lon in config.LAVA_TUBE_SITES)
    feature = config.SUBSURFACE_TARGETS["moon_mare_tranq_pit"]
    assert feature.temperature_k == 290.0 and "Horvath" in feature.source


def test_subsurface_takes_the_annual_mean_of_the_nearest_cell(monkeypatch):
    from src import sources

    grid_lat, grid_lon = sources.POWER_STORES["merra2"][1:]
    field = np.zeros((13, grid_lat.size, grid_lon.size), dtype=np.float32)
    field[12] = np.broadcast_to(grid_lat[:, None], field[12].shape)       # annual value = latitude of the cell, in C
    monkeypatch.setattr(sources, "power_climatology", lambda store, variable: field)
    table = data_prep.build_subsurface()
    assert list(table.columns) == list(data_prep.SUBSURFACE_COLUMNS) and len(table) == len(config.LAVA_TUBE_SITES)
    expected = np.round(table["lat"] * 2) / 2 + 273.15                    # MERRA-2 latitudes are 0.5 deg apart
    assert np.allclose(table["maat_k"], expected, atol=0.01)


def test_subsurface_round_trip_with_the_dataset(tmp_path):
    ds = synthetic.build_dataset(n_clusters=20, tiles_per_cluster=10, seed=1, with_zoom=True)
    assert ds.subsurface is not None and (ds.subsurface["source"] == "Invented temperature (test data)").all()
    back = data_prep.load_dataset(data_prep.save_dataset(ds, tmp_path / "layers"))
    pd.testing.assert_frame_equal(back.subsurface, ds.subsurface)
    assert synthetic.build_dataset(n_clusters=20, tiles_per_cluster=10, seed=1).subsurface is None


# ------------------------------------------------ Mars thermal profile
def test_diurnal_cycle_thermal_takes_amplitude_and_minimum_per_season():
    hours = np.arange(25)
    shape = np.sin((hours - 7) / 24 * 2 * np.pi)
    cycles = np.stack([200.0 + 10.0 * shape, 230.0 + 40.0 * shape])       # a cold flat season, a warm wide one
    q = data_prep.diurnal_cycle_thermal(cycles, np.array([1.0, 1.0]))
    assert q.shape == (2, config.QUANTILE_LEVELS.size)
    assert q[0].min() >= 20.0 - 1e-3 and q[0].max() <= 80.0 + 1e-3         # amplitudes: 20 K and 80 K
    assert q[1].min() >= 190.0 - 1e-3 and q[1].max() <= 190.0 + 1e-3       # both seasons bottom out at 190 K
    assert (np.diff(q, axis=1) >= -1e-5).all()
    mostly_cold = data_prep.diurnal_cycle_thermal(cycles, np.array([9.0, 1.0]))
    assert np.median(mostly_cold[0]) < np.median(q[0])                     # time weights count


def test_in_situ_control_quotes_the_published_range():
    text = data_prep._in_situ_control("mars_jezero", "temp_min_k", np.array([188.0, 201.0]))
    low, high, _ = config.TARGET_REFERENCE_VALUES["mars_jezero"]["temp_min_k"]
    assert f"{low:.0f}-{high:.0f} K" in text and "188-201 K" in text
    assert data_prep._in_situ_control("moon_shackleton", "temp_min_k", np.array([40.0])) == ""


def test_horizontal_irradiance_is_never_negative():
    """Regression (14 Nov 2026): on a summit the Sun is visible below the horizontal plane; the insolation of
    Malapert had a negative 5th percentile (-0.84 W/m2)."""
    sin_e = np.sin(np.radians([-1.0, 2.0, 5.0, -3.0]))
    visible = np.array([True, True, False, False])       # depressed horizon: visible at -1 deg
    factor = data_prep.horizontal_irradiance_factor(sin_e, visible)
    assert (factor >= 0).all() and factor[0] == 0 and factor[2] == 0
    assert factor[1] == pytest.approx(np.sin(np.radians(2.0)))


# ------------------------------------------------------------ height of the Sun
def test_sun_elevation_follows_latitude_and_tilt():
    equator = data_prep.sun_elevation_quantiles(0.0, "Earth")
    arctic = data_prep.sun_elevation_quantiles(80.0, "Earth")
    assert (np.diff(equator) >= 0).all() and equator[0] > 0
    assert equator[-1] > 70 and arctic[-1] < 90 - 80 + 23.44 + 0.1        # never above 90 - lat + tilt
    assert arctic[9] < equator[9]
    moon_pole = data_prep.sun_elevation_quantiles(-89.67, "Moon")
    assert moon_pole[-1] < 90 - 89.67 + 1.54 + 0.05                       # a grazing Sun: below 1.9 deg
    values, weights = data_prep.sun_elevation_samples(45.0, "Mars")
    assert weights.min() == 0 and (values[weights > 0] > 0).all()         # only daytime samples weigh


def test_window_latitudes_span_the_window():
    lats = data_prep.window_latitudes(-89.67, 55.0, data_prep.BODY_RADIUS_KM["Moon"])
    assert lats.size == 81 and lats.min() < -89.6 and -87.5 < lats.max() < -86.9
    flat = data_prep.window_latitudes(18.0, 55.0, data_prep.BODY_RADIUS_KM["Mars"])
    assert flat.min() == pytest.approx(18.0 - np.degrees(55.0 / 3389.5), abs=0.01)


def test_monthly_sun_elevation_has_a_polar_night():
    monthly = data_prep.earth_monthly_sun_elevation(80.0)
    assert monthly.shape == (12, config.QUANTILE_LEVELS.size)
    assert np.isnan(monthly[11]).all() and np.isfinite(monthly[5]).all()   # December dark, June lit
    assert monthly[5, 9] > monthly[2, 9]                                    # the Sun is higher in June than in March
    tropics = data_prep.earth_monthly_sun_elevation(10.0)
    assert np.isfinite(tropics).all()


def test_every_target_declares_its_ice_and_the_new_variable(small_ds):
    assert set(t["target_id"] for t in data_prep.REAL_TARGETS) == set(config.TARGET_NEAR_SURFACE_ICE)
    assert "sun_elevation_deg" in small_ds.variables
    assert small_ds.sigma_default[list(small_ds.variables).index("sun_elevation_deg")] > 0


# ------------------------------------------------------------ place names (src/places.py)
def test_points_in_polygon_handles_holes_and_edges():
    from src import places

    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    hole = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
    lon = np.array([0.5, 2.0, 5.0, 3.5, -1.0])
    lat = np.array([0.5, 2.0, 2.0, 3.5, 2.0])
    assert places.points_in_polygon(lon, lat, [square]).tolist() == [True, True, False, True, False]
    assert places.points_in_polygon(lon, lat, [square, hole]).tolist() == [True, False, False, True, False]


def test_name_points_prefers_the_most_specific_feature():
    from src import places

    def box(west, south, east, north):
        return [[np.array([[west, south], [east, south], [east, north], [west, north]])]]

    features = [("Big Plateau", box(0, 0, 10, 10)), ("Small Range", box(2, 2, 4, 4)), ("", box(0, 0, 20, 20))]
    names = places.name_points(np.array([3.0, 8.0, 15.0]), np.array([3.0, 8.0, 15.0]), features)
    assert names.tolist() == ["Small Range", "Big Plateau", ""]
    # both names of a tile are read at one point: the centre, else the first nearby point with a primary name
    countries = [("Westland", box(0, 0, 10, 10)), ("Eastland", box(11, 0, 20, 10))]
    regions = [("Western Hills", box(0, 0, 10.9, 10)), ("Eastern Range", box(11, 0, 20, 10))]
    lat, lon = np.array([5.0, 5.0, 5.0, 50.0]), np.array([5.0, 10.2, 10.75, 50.0])
    country, region = places.name_tiles(lat, lon, countries, regions)
    assert country.tolist() == ["Westland", "Westland", "Eastland", ""]     # centres 2 and 3 lie in the strait
    # the third centre is in the 'Western Hills' polygon, but its country comes from the eastern shore:
    assert region.tolist() == ["Western Hills", "Western Hills", "Eastern Range", ""]


def test_place_labels_and_site_names():
    from src import places, report

    assert places.clean_name("ELLESMERE  ISLAND") == "Ellesmere Island"
    assert places.clean_name("Altun Mts.") == "Altun Mts."
    assert places.place_label("China", "Kunlun Mountains") == "Kunlun Mountains, China"
    assert places.place_label("", "Karakoram") == "Karakoram"                      # disputed area: no country
    assert places.place_label("Greenland", "Greenland") == "Greenland" and places.place_label("", "") == ""
    table = pd.DataFrame({"lat": [1.5, 2.5], "lon": [3.5, 4.5], "country": ["Chile", ""], "region": ["", ""]})
    assert places.tile_labels(table, 2).tolist() == ["Chile", ""]
    assert places.tile_labels(None, 3).tolist() == ["", "", ""] and places.tile_labels(table, 5).tolist() == [""] * 5
    assert report.site_name("Atacama Desert, Chile", -24.5, -69.5) == "Atacama Desert, Chile (24.5 S, 69.5 W)"
    assert report.site_name("", -24.5, -69.5) == "24.5 S, 69.5 W"


def test_places_round_trip_and_mismatch(tmp_path, small_ds):
    """places.csv is read back only when it lists the tiles of the dataset, in order."""
    ds = dataclasses.replace(small_ds)
    ds.places = pd.DataFrame({"lat": ds.tile_lat.astype(float), "lon": ds.tile_lon.astype(float),
                              "country": ["Chile"] * ds.n_tiles, "region": [""] * ds.n_tiles})
    out = data_prep.save_dataset(ds, tmp_path / "with_places")
    back = data_prep.load_dataset(out)
    assert back.places is not None and back.places["country"].iloc[0] == "Chile" and back.places["region"].iloc[0] == ""
    ds.places = ds.places.iloc[:-1]
    assert data_prep.load_dataset(data_prep.save_dataset(ds, tmp_path / "short")).places is None
