"""Tests of the figure helpers (pure numpy parts; figures only when plotly is installed)."""
import numpy as np
import pytest

from src import charts


def test_hillshade_of_flat_and_tilted_ground():
    flat = charts.hillshade(np.zeros((6, 6)), 100.0, 100.0, altitude_deg=30.0)
    assert np.allclose(flat, 0.5)                                    # sin(30 deg): light on a horizontal surface
    x = np.arange(8, dtype=float) * 100.0
    rises_east = np.tile(x, (8, 1))                                  # 45 deg slope facing west
    lit_from_west = charts.hillshade(rises_east, 100.0, 100.0, azimuth_deg=270.0, altitude_deg=45.0)
    lit_from_east = charts.hillshade(rises_east, 100.0, 100.0, azimuth_deg=90.0, altitude_deg=45.0)
    assert np.allclose(lit_from_west, 1.0) and np.allclose(lit_from_east, 0.0)
    rises_north = np.tile(x[::-1, None], (1, 8))                     # row 0 is north
    assert np.allclose(charts.hillshade(rises_north, 100.0, 100.0, azimuth_deg=180.0, altitude_deg=45.0), 1.0)


def test_hillshade_keeps_missing_data_and_row_wise_pixels():
    dem = np.zeros((5, 5))
    dem[2, 2] = np.nan
    shade = charts.hillshade(dem, 100.0, np.linspace(90.0, 110.0, 5))
    assert np.isnan(shade[2, 2]) and np.isfinite(shade[0, 0])


@pytest.mark.skipif(not charts.HAS_PLOTLY, reason="plotly not installed")
def test_zoom_figure_has_three_panels_on_one_scale():
    rng = np.random.default_rng(0)
    site = rng.normal(size=(48, 24))
    bounds = np.array([[0.5, 1.0, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0]])   # two sub-areas stacked north / south
    fig = charts.zoom_figure(site, (463.0, np.full(48, 463.0)), rng.normal(size=(60, 60)), bounds,
                             np.array([0.8, np.nan]), (2, 1), best=0)
    assert len(fig.data) == 3 and len(fig.layout.shapes) == 2
    ranges = {tuple(fig.layout[axis].range) for axis in ("xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3")}
    assert len(ranges) == 1                                           # same ground scale everywhere
    no_target = charts.zoom_figure(site, (463.0, np.full(48, 463.0)), None, bounds, np.array([0.8, 0.1]), (2, 1))
    assert len(no_target.data) == 2


@pytest.mark.skipif(not charts.HAS_PLOTLY, reason="plotly not installed")
def test_atlas_map_has_one_trace_per_target():
    lat, lon = np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0])
    fig = charts.atlas_map(lat, lon, np.array([0, 1, 1]), np.array([0.9, 0.8, 0.7]), ["A", "B"], selected=(5.0, 5.0))
    assert [t.name for t in fig.data] == ["A (1)", "B (2)", "Selected place"]
    assert list(fig.data[1].lat) == [20.0, 30.0] and list(fig.data[1].customdata) == [80.0, 70.0]


def test_radar_draws_the_threshold_per_variable():
    """The strong-match line follows the per-variable thresholds; a circle at 0.8 contradicted the statuses."""
    pytest.importorskip("plotly")
    labels = ["a", "b", "c"]
    fig = charts.radar(labels, np.array([0.9, 0.5, 0.2]), strong=np.array([0.95, 0.6, 0.3]))
    assert list(fig.data[0].r) == [0.95, 0.6, 0.3, 0.95]
    assert len(set(charts.radar(labels, np.array([0.9, 0.5, 0.2])).data[0].r)) == 1      # fallback: one circle
