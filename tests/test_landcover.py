"""Tests of the bare-ground variable: projection, rock outcrop raster, WorldCover classes."""
import numpy as np
import pytest

from src import landcover


def test_polar_stereographic_inverse_matches_gdal():
    from rasterio.warp import transform

    rng = np.random.default_rng(0)
    x, y = rng.uniform(-2.7e6, 2.7e6, 200), rng.uniform(-2.5e6, 2.5e6, 200)
    lon_ref, lat_ref = transform("EPSG:3031", "EPSG:4326", x.tolist(), y.tolist())
    lon, lat = landcover.polar_stereographic_south_to_lonlat(x, y)
    assert np.allclose(lat, lat_ref, atol=1e-9)
    assert np.allclose((lon - np.asarray(lon_ref) + 180) % 360 - 180, 0.0, atol=1e-9)
    assert landcover.polar_stereographic_south_to_lonlat(np.array([0.0]), np.array([0.0]))[1][0] == -90.0


def test_rock_outcrop_rasterizes_a_square_and_crosses_the_antimeridian():
    square = np.array([[10.25, -70.75], [10.75, -70.75], [10.75, -70.25], [10.25, -70.25], [10.25, -70.75]])
    across = np.array([[179.5, -80.6], [-179.5, -80.6], [-179.5, -80.4], [179.5, -80.4], [179.5, -80.6]])
    rock = landcover.RockOutcrop([[square], [across]])
    assert rock.rasterize(-71, 10, (40, 40)).mean() == pytest.approx(0.25)
    assert not rock.rasterize(-72, 10, (40, 40)).any()
    east, west = rock.rasterize(-81, 179, (40, 40)), rock.rasterize(-81, -180, (40, 40))
    assert east[:, 20:].any() and not east[:, :20].any()         # the polygon touches 180 deg from the west...
    assert west[:, :20].any() and not west[:, 20:].any()         # ...and from the east, not across the tile


def test_bare_fraction_from_worldcover_classes():
    classes = np.full((10, 10), 10, dtype=np.uint8)              # tree cover
    classes[:5, :5] = 60                                         # bare / sparse vegetation
    classes[5:, :5] = 80                                         # permanent water
    classes[5:, 5:8] = 60
    classes[5:, 8:] = 80                                         # 60 % bare, 40 % water
    frac = landcover.bare_fraction_from_classes(classes, (2, 2))
    assert frac[0, 0] == 1.0 and frac[0, 1] == 0.0
    assert np.isnan(frac[1, 0])                                  # mostly water: not land
    assert frac[1, 1] == pytest.approx(1.0)                      # bare share of the land part
