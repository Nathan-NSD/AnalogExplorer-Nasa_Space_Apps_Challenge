# Data: sources, licences, rebuilding, limits and reproducibility

Every value in AnalogExplorer comes from open data. The reading code is in `src/sources.py`, the construction of
the dataset in `src/data_prep.py`, and the provenance of every target value is recorded in the dataset's
`meta.json`. The application never shows invented data: without its dataset it stops and says how to build it.

## 1. Sources by variable

Each variable is available for the Earth, the Moon and Mars, or is fixed by physics for the targets (for
example, there is no liquid water and no vegetation on the Moon); its source is written in `src/config.py`.

| Variable | Earth (tiles) | Moon (targets) | Mars (targets) |
|---|---|---|---|
| Slope, roughness | Copernicus DEM GLO-90 resampled to ~463 m, water masked (water body mask) | LRO LOLA: 240 m polar DEM, LDEM_64 (~474 m) | MGS MOLA MEGDR 128 pixels per degree (~463 m) |
| Diurnal temperature amplitude | NASA POWER 2 × `TS_AMP` (mean daily maximum - mean daily minimum) | LRO Diviner Global Cumulative Products: max - min of bolometric temperature | Mars Climate Database v6.2: max - min of the diurnal cycle of surface temperature, 12 seasons |
| Minimum temperature | NASA POWER `TS_MIN_AVG` (mean daily minimum of surface temperature, the counterpart of the minimum of the targets' mean cycle) | LRO Diviner: minimum of bolometric temperature | Mars Climate Database: coldest hour of the cycle |
| Aridity | 1 - min(P/PET, 1) (UNEP aridity index), P = `PRECTOTCORR`, PET from Hargreaves; months above 0 °C; 1 when water is never liquid | 1 (physics) | 1 (physics) |
| Ground below 0 °C at depth (thermal permafrost: a temperature, not a proof of ice) | share of months with the soil at 3-10 m below 0 °C (`TSOIL6`) | 1 (physics) | 1 (physics) |
| Mean insolation (W/m², horizontal) | NASA POWER `ALLSKY_SFC_SW_DWN` | orbit (equator); LOLA horizons (poles) | orbit × atmospheric transmission 0.75 |
| Sunlit fraction | astronomical day length × (1 - `CLOUD_AMT_DAY`) | 0.5 (equator); NASA PGDA illumination map (poles) | astronomical day length (no clouds) |
| Sun elevation in daytime | astronomy: latitude of the tile centre, obliquity 23.44°, orbit; no relief, no clouds | astronomy: latitudes of the target window, obliquity 1.54° | astronomy: latitudes of the window, obliquity 25.19°, seasons weighted by their length |
| Bare ground | share of each ~463 m pixel without vegetation, snow or ice: ESA WorldCover 10 m (class 60) north of 60°S, SCAR rock outcrops south of it; 25 samples per pixel, water excluded | 1 (physics) | 1 (physics) |

## 2. Provenance, access and licences

All sources are read without an account.

| Source | Access | Licence and attribution |
|---|---|---|
| NASA POWER climatology (MERRA-2 0.5° × 0.625°; CERES SYN1deg 1°) | public AWS bucket `nasa-power` (Zarr) | NASA open data. Data obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program. |
| Copernicus DEM GLO-90 and water body masks | public AWS bucket `copernicus-dem-90m` (Cloud Optimized GeoTIFF, read remotely) | Copernicus DEM licence, free with attribution: produced using Copernicus WorldDEM-90 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. |
| LRO LOLA (240 m polar GDR, LDEM_64) | NASA PDS Geosciences Node | NASA data (public domain); Smith et al. 2010 |
| LRO Diviner Global Cumulative Products | NASA PDS Geosciences Node | NASA data; Paige et al. 2010 |
| MGS MOLA MEGDR 128 ppd | NASA PDS Geosciences Node (windows read remotely) | NASA data; Smith et al. 2001 |
| South-pole average illumination `AVGVISIB_75S_120M_201608.TIF` | NASA Planetary Geodesy Data Archive (PGDA) | NASA data; Mazarico et al. 2011 |
| ESA WorldCover 10 m 2021 v200 (land cover, 60°S to 84°N) | public AWS bucket `esa-worldcover` | CC BY 4.0. © ESA WorldCover project 2021 / contains modified Copernicus Sentinel data (2021) processed by the ESA WorldCover consortium; Zanaga et al. 2022 |
| Antarctic rock outcrops, high resolution v7.11 (Sentinel-2 2024/25) | SCAR Antarctic Digital Database, UK Polar Data Centre | CC BY 4.0. Data from the SCAR Antarctic Digital Database, 2025, doi:10.5285/815525ca-cefe-4fbf-9224-3ee784e7de4e |
| Mars surface temperature: Mars Climate Database v6.2 (LMD, Open University, IAA, with ESA and CNES) | web interface, 48 requests | Millour, E., et al. (The MCD Team): The Latest Mars Climate Database, MCD version 6.2, Europlanet Science Congress 2026, EPSC2026-776, doi:10.5194/epsc2026-776; Forget et al. 1999 |
| Place names: Natural Earth v5.1.2 (countries 1:50m, physical regions 1:10m, disputed areas) | GitHub `nvkelso/natural-earth-vector` | public domain. Made with Natural Earth. |
| Coordinates of the catalogue of known analogs | Wikipedia and published tables (the `source` field of each site) | Wikipedia: CC BY-SA 4.0 |
| Martian in situ temperature ranges (cross-check only) | published papers cited in `config.TARGET_REFERENCE_VALUES` | citation of the papers |
| Subsurface case study: Mare Tranquillitatis pit | LROC Lunar Pit Atlas; Horvath et al. 2022 | NASA/GSFC/Arizona State University; citation of the paper |
| Subsurface case studies: lava tubes on Earth | Wikipedia coordinates; NASA POWER annual air temperature `T2M` | Wikipedia: CC BY-SA 4.0; POWER as above |

## 3. Running and rebuilding

- **To run the app**: nothing to download. The dataset `data/processed/real_v1/` (~16 MB) is in the repository.
- **To rebuild it**: `python -m src.data_prep --real` (about one hour the first time; quick trial:
  `--real --limit 300`). The script downloads what it needs into reusable caches:
  - `data/raw/power/`: POWER variables (~60 MB);
  - `data/raw/diviner/`: two Diviner bands (~312 MB) and their processed version;
  - `data/raw/lola/`: 240 m polar DEM (58 MB);
  - `data/raw/copernicus/tileList.txt`; the DEM tiles themselves (~4 GB transferred) are read and not stored;
  - `data/processed/cache/topography_wbm/`: topography quantiles per tile (resumable);
  - `data/raw/landcover/`: WorldCover grid and SCAR rock outcrops (~290 MB); WorldCover tiles are read
    remotely; `data/processed/cache/bare_ground/`: bare-ground quantiles per tile (resumable).
- MOLA, LDEM_64 and PGDA windows are read remotely; nothing is stored.
- **Optional layers** (fine-scale zoom, subsurface case studies) are computed at the end of `--real`;
  `python -m src.data_prep --layers` adds them to an existing dataset (~1 min).
- **Place names only**: `python -m src.data_prep --places` (a few seconds).
- **Targets only**: `python -m src.data_prep --targets` recomputes the target profiles (including the Martian
  thermal cycle, cached in `data/raw/mcd/`) and the layers, without reading the Earth tiles again (~6 min).

## 4. Data contract

One folder per dataset (`data/processed/<name>/`); the app and the written atlas read `real_v1/`, or the folder
named by the environment variable `ANALOG_DATASET_DIR`:
- `tiles.npz`: `tile_id, lat, lon, valid_frac`, quantiles `(n_tiles, n_vars, 19)`, quantile levels, default
  tolerances σ (pairs of places drawn in proportion to land area), target quantiles `(n_targets, n_vars, 19)`
  (a NaN row means the variable is not evaluated for that target); `n_vars` = 10 and `n_targets` = 7;
  `valid_frac` (land share of the tile) enters the land-area weight of each tile;
- `targets.csv`: `target_id, name, body, lat, lon, note`;
- `known_analogs.csv`: `name, lat, lon, relevant_to, tested_mars, tested_moon, extent_km, source` (tested
  properties separated by `|`);
- `meta.json`: source, version, creation date, variables, resolutions, sources and provenance of every target
  value;
- `zoom.npz` (optional, ~5 MB): elevation (int16, display only), slope and roughness quantiles of each
  ~11 km sub-area, land mask and grid of each zoomed tile, and the elevation window of each target;
- `places.csv` (optional): `lat, lon, country, region`, one row per tile in the order of `tiles.npz`; display
  only, never used in a calculation; `country` is empty in disputed areas;
- `subsurface.csv` (optional): `name, country, lat, lon, maat_k, source`, lava tubes on Earth with the mean
  annual air temperature of their POWER cell.

The similarities to a target are computed when the dataset is loaded (one matrix product) and then cached.

## 5. Known limits

- Earth's climate comes from grids of 0.5° to 1°: a tile holds 1 to 4 POWER cells, so its climate distribution
  mostly reflects the 12 months.
- POWER temperatures are monthly means of skin temperature, smoother than an in situ measurement; Diviner gives a
  bolometric temperature: close but not identical quantities.
- Hargreaves PET is an empirical formula (FAO-56); frozen months are excluded from aridity, and a tile that is
  always frozen scores 1 by convention (no liquid water), like the Moon and Mars.
- Mars: the thermal cycle comes from a climate model (Mars Climate Database), at one point at the centre of each
  target, without a global dust storm; in situ ranges are used as a cross-check only. The Arcadia Planitia window
  is placed approximately.
- Sun elevation: smooth sphere, no relief and no refraction. At the lunar poles the Sun stays below 3 to 6°
  (95th percentile), which no place on Earth reproduces over the year (at best 1.4 to 1.6 σ away); the site
  sheet's "When to go" calendar gives the months that come closest. Calendar months are placed with the mean
  motion of the Sun (within ~2° of solar longitude).
- Land-area weight: cos(latitude of the centre) × land share of the tile; a coastal tile counts for its land part.
- Place names: name of the tile centre (or of a nearby point for a coastal tile), not of the whole tile; de facto
  boundaries of Natural Earth, no country in disputed areas; 482 tiles have no name.
- Catalogue of known analogs: coordinates from Wikipedia, sites often smaller than a tile; each is matched to the
  nearest tile within 150 km.
- Topography: one common scale of ~463 m for the three worlds, rather than one scale per target.
- Bare ground: two sources joined at 60°S (WorldCover 2021, SCAR 2024/25); "bare" includes sparse vegetation
  (< 10 %). ESA CCI Land Cover was not used: it classifies the whole of Antarctica as ice, dry valleys included.
  Coastal Antarctic pixels count the sea in their denominator (slight downward bias). The target value is 1 by
  physics (no vegetation, no snow or ice cover). WorldCover has no data north of 83°N, so 14 tiles of northern
  Greenland are left out (23,039 tiles). 71 % of the tiles (65 % of the land) have no bare pixel: for such a
  bounded variable, the tolerance is P(two places differ) × the median gap of the pairs that differ.

## 6. Reproducibility

- Pinned versions in `requirements.txt` (app) and `requirements-dev.txt` (tests, lint, offline pipeline).
  Python 3.12 or later for the app; 3.14 or later to rebuild the dataset (the NASA POWER reader uses `compression.zstd`).
- Fixed seeds: tolerances, robustness, score interval and credibility check (`seed=0`); synthetic test
  dataset (`42`).
- `meta.json` dates each dataset and traces the provenance of every target value.
- The continuous-integration workflow runs the linter and the tests on Python 3.12 and 3.15 at every push.
