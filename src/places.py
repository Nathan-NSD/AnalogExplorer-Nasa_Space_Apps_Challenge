"""Place names of the Earth tiles: the country and the physical region of each tile.

Display only: a name never enters the score. Source: Natural Earth (public domain), countries at 1:50m and
named physical regions (ranges, deserts, plateaus, islands) at 1:10m, read once by the offline pipeline
(`python -m src.data_prep --places`) and stored next to the dataset (places.csv). Pure numpy: no GIS library.
Country lines are Natural Earth's de facto boundaries; they are a reading aid, not a statement on borders:
a tile whose centre lies in an area Natural Earth lists as disputed or breakaway gets no country at all.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

NATURAL_EARTH_RELEASE = "v5.1.2"
NATURAL_EARTH_BASE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                      f"{NATURAL_EARTH_RELEASE}/geojson")
COUNTRIES_LAYER = "ne_50m_admin_0_countries"
REGIONS_LAYER = "ne_10m_geography_regions_polys"
DISPUTED_LAYER = "ne_50m_admin_0_breakaway_disputed_areas"
NATURAL_EARTH_CITATION = f"Natural Earth {NATURAL_EARTH_RELEASE} (public domain), naturalearthdata.com"
PLACES_FILE = "places.csv"
PLACES_COLUMNS = ("lat", "lon", "country", "region")
# Region classes that do not name a place on land precisely enough ("Dragons-be-here" is Natural Earth's own
# placeholder class).
SKIPPED_REGION_CLASSES = ("Continent", "Dragons-be-here", "Lake")
# A tile is named from its centre; when the centre has no country (sea, gap between polygons), from the first
# of four points 0.3 deg away that has one (the tile is 1 x 1 deg).
SAMPLE_OFFSETS = ((0.0, 0.0), (0.3, 0.3), (0.3, -0.3), (-0.3, 0.3), (-0.3, -0.3))

Polygon = list[np.ndarray]          # rings of (lon, lat) vertices: the outline, then the holes


def points_in_polygon(lon: np.ndarray, lat: np.ndarray, rings: Sequence[np.ndarray]) -> np.ndarray:
    """(n,) bool: points inside a polygon given by its rings, by the even-odd rule (holes are excluded).

    rings: arrays (v, 2) of (lon, lat) vertices, closed or not. A ray toward +lon is cast from every point.
    """
    x, y = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    inside = np.zeros(x.shape, dtype=bool)
    for ring in rings:
        r = np.asarray(ring, dtype=np.float64)
        x0, y0 = r[:, 0], r[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        for ax, ay, bx, by in zip(x0, y0, x1, y1, strict=True):
            if ay == by:
                continue                                  # a horizontal edge never crosses the ray
            crosses = (ay > y) != (by > y)
            at = ax + (y - ay) * (bx - ax) / (by - ay)
            inside ^= crosses & (x < at)
    return inside


def geometry_polygons(geometry: dict) -> list[Polygon]:
    """Polygons of a GeoJSON Polygon or MultiPolygon geometry, each as its list of rings."""
    kind, coords = geometry["type"], geometry["coordinates"]
    if kind == "Polygon":
        coords = [coords]
    elif kind != "MultiPolygon":
        raise ValueError(f"unsupported geometry {kind!r}")
    return [[np.asarray(ring, dtype=np.float64)[:, :2] for ring in polygon] for polygon in coords]


def clean_name(text: str | None) -> str:
    """'ELLESMERE  ISLAND' -> 'Ellesmere Island'; names already in mixed case are only stripped."""
    name = re.sub(r"\s+", " ", str(text or "")).strip()
    return name.title() if name.isupper() else name


def name_points(lon: np.ndarray, lat: np.ndarray, features: Sequence[tuple[str, Sequence[Polygon]]]) -> np.ndarray:
    """(n,) name of the most specific feature holding each point ('' when none does).

    features: (name, polygons). When several features hold a point (a range inside a plateau inside a
    geographic area), the one with the smallest bounding box wins.
    """
    x, y = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    names = np.full(x.shape, "", dtype=object)
    best_area = np.full(x.shape, np.inf)
    for name, polygons in features:
        if not name:
            continue
        outlines = np.concatenate([p[0] for p in polygons])
        area = float(np.ptp(outlines[:, 0]) * np.ptp(outlines[:, 1]))
        hit = np.zeros(x.shape, dtype=bool)
        for rings in polygons:
            west, south = rings[0].min(axis=0)
            east, north = rings[0].max(axis=0)
            near = np.flatnonzero((x >= west) & (x <= east) & (y >= south) & (y <= north))
            if near.size:
                hit[near[points_in_polygon(x[near], y[near], rings)]] = True
        better = hit & (area < best_area)
        names[better], best_area[better] = name, area
    return names


def name_tiles(tile_lat: np.ndarray, tile_lon: np.ndarray, primary: Sequence[tuple[str, Sequence[Polygon]]],
               secondary: Sequence[tuple[str, Sequence[Polygon]]]) -> tuple[np.ndarray, np.ndarray]:
    """(n,) primary and secondary name of each tile (e.g. country and physical region), read at the same point.

    The point is the tile centre when the primary layer names it, else the first point of SAMPLE_OFFSETS that it
    names (coastal tiles, whose centre is at sea). Both names come from that one point, so a strait between two
    lands cannot get the country of one shore and the region of the other. A tile without any primary name
    keeps the secondary name of its centre.
    """
    lat, lon = np.asarray(tile_lat, dtype=np.float64), np.asarray(tile_lon, dtype=np.float64)
    first = np.stack([name_points(lon + dx, lat + dy, primary) for dy, dx in SAMPLE_OFFSETS])      # (5, n)
    second = np.stack([name_points(lon + dx, lat + dy, secondary) for dy, dx in SAMPLE_OFFSETS])
    named = first != ""
    point = np.where(named.any(axis=0), named.argmax(axis=0), 0)          # first sample with a primary name
    columns = np.arange(lat.size)
    return first[point, columns], second[point, columns]


def read_layer(path: str | Path, name_field: str, skipped_classes: Sequence[str] = ()) -> list:
    """(name, polygons) features of a Natural Earth GeoJSON layer; features of `skipped_classes` are left out."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    features = []
    for feature in data["features"]:
        props = feature["properties"]
        if props.get("FEATURECLA") in skipped_classes or feature.get("geometry") is None:
            continue
        features.append((clean_name(props.get(name_field)), geometry_polygons(feature["geometry"])))
    return features


def build_places(tile_lat: np.ndarray, tile_lon: np.ndarray, log=print) -> pd.DataFrame:
    """Country and physical region of every tile, from Natural Earth (downloads two layers once)."""
    from src import sources

    def layer(name: str) -> Path:
        return sources.cached_download(f"{NATURAL_EARTH_BASE}/{name}.geojson",
                                       sources.RAW_DIR / "natural_earth" / f"{name}.geojson")

    country, region = name_tiles(tile_lat, tile_lon, read_layer(layer(COUNTRIES_LAYER), "NAME"),
                                 read_layer(layer(REGIONS_LAYER), "NAME_EN", SKIPPED_REGION_CLASSES))
    disputed = name_points(np.asarray(tile_lon), np.asarray(tile_lat), read_layer(layer(DISPUTED_LAYER), "NAME")) != ""
    country[disputed] = ""                                  # no country is shown for a disputed area
    log(f"Places: {int((country != '').sum())} of {len(country)} tiles with a country ({int(disputed.sum())} in "
        f"disputed areas left without one), {int((region != '').sum())} with a named region.")
    return pd.DataFrame({"lat": np.asarray(tile_lat, dtype=np.float64), "lon": np.asarray(tile_lon, dtype=np.float64),
                         "country": country, "region": region}, columns=list(PLACES_COLUMNS))


def place_label(country: str | None, region: str | None) -> str:
    """'Kunlun Mountains, China', 'China', 'Kunlun Mountains' or '' from a tile's two names."""
    parts = [p for p in (clean_name(region), clean_name(country)) if p and p.lower() != "nan"]
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():      # e.g. region 'Greenland', country 'Greenland'
        parts = parts[:1]
    return ", ".join(parts)


def tile_labels(places: pd.DataFrame | None, n_tiles: int) -> np.ndarray:
    """(n_tiles,) display name of each tile ('' without the layer or without a name)."""
    if places is None or len(places) != n_tiles:
        return np.full(n_tiles, "", dtype=object)
    return np.array([place_label(c, r) for c, r in zip(places["country"].fillna(""), places["region"].fillna(""),
                                                       strict=True)], dtype=object)
