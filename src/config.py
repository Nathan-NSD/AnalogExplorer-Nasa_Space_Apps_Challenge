"""Project-wide constants and domain definitions.

This module holds *definitions* only (no heavy logic):
- the variables compared between Earth tiles and a planetary target,
- the mission profiles and their rationale,
- the rule bases used by the report (test activities, structural limits),
- default engine parameters (all of them are starting values to calibrate).

data_prep.py produces the variables of VARIABLES in the same order (data contract: docs/data.md, 4);
every constant carries its source or the words 'hypothesis to calibrate'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Variable:
    """One physical quantity compared between worlds."""

    name: str
    label: str
    dimension: str
    unit: str
    description: str


VARIABLES: tuple[Variable, ...] = (
    Variable("slope_deg", "Slope", "Topography", "deg",
             "Local slope on a common ground sampling distance."),
    Variable("roughness_m", "Roughness", "Topography", "m",
             "Local standard deviation of elevation in a fixed window."),
    Variable("diurnal_amp_k", "Diurnal temperature amplitude", "Thermal", "K",
             "Day/night surface temperature range."),
    Variable("temp_min_k", "Minimum temperature", "Thermal", "K",
             "Coldest surface temperature of the cycle."),
    Variable("aridity", "Aridity index", "Water / Ice", "0-1",
             "1 - P/PET (UNEP aridity index) in months above 0 C: 0 = humid, 1 = hyperarid or no liquid water."),
    Variable("ice_proxy", "Ground below 0 C at depth", "Water / Ice", "0-1",
             "Share of the year the ground stays below 0 C at 3-10 m (thermal permafrost = 1): a temperature "
             "condition, not proof of ice."),
    Variable("insolation_wm2", "Mean insolation", "Illumination", "W/m2",
             "Annual mean solar irradiance reaching a horizontal surface (atmosphere and terrain included)."),
    Variable("sunlit_frac", "Sunlit fraction", "Illumination", "0-1",
             "Fraction of total time with direct sunlight (terrain shadows and clouds included)."),
    Variable("sun_elevation_deg", "Sun elevation in daytime", "Illumination", "deg",
             "Height of the Sun above the horizon while it is up, over the year: astronomy only (latitude and "
             "axial tilt, smooth sphere), so terrain shadows and clouds are not in it."),
    Variable("bare_frac", "Bare ground", "Surface cover", "0-1",
             "Share of the ground that is bare rock or soil, without vegetation, permanent snow or ice."),
)

# Where each variable comes from on each world: (Earth tiles, Moon targets, Mars targets). Shown on the home page
# so that the chain data -> variable -> dimension -> score is visible; details, licences and citations are in
# docs/data.md, 1-2 (source of truth).
_DEM_SOURCES = ("Copernicus DEM GLO-90 (ESA), read at ~463 m, water masked",
                "LRO LOLA (NASA PDS): polar DEM 240 m, LDEM_64", "MGS MOLA MEGDR 128 ppd (NASA PDS)")
_BY_PHYSICS = "1, set by physics"
VARIABLE_SOURCES: dict[str, tuple[str, str, str]] = {
    "slope_deg": _DEM_SOURCES,
    "roughness_m": _DEM_SOURCES,
    "diurnal_amp_k": ("NASA POWER (MERRA-2): mean daily max - mean daily min of the surface temperature",
                      "LRO Diviner (NASA PDS): max - min of the bolometric temperature",
                      "Mars Climate Database v6.2 (LMD, with ESA and CNES): diurnal cycle of the surface temperature"),
    "temp_min_k": ("NASA POWER (MERRA-2): mean daily minimum of the surface temperature",
                   "LRO Diviner: coldest bolometric temperature", "Mars Climate Database: coldest hour of the cycle"),
    "aridity": ("NASA POWER (MERRA-2): precipitation over potential evapotranspiration (UNEP aridity index)",
                f"{_BY_PHYSICS} (vacuum: no liquid water)", f"{_BY_PHYSICS} (pressure near the triple point)"),
    "ice_proxy": ("NASA POWER (MERRA-2): soil temperature at 3-10 m", f"{_BY_PHYSICS} (Diviner: far below 273 K)",
                  f"{_BY_PHYSICS} (mean ground temperature below 240 K)"),
    "insolation_wm2": ("NASA POWER (CERES SYN1deg): all-sky shortwave irradiance",
                       "Orbit (equator); horizon method on LRO LOLA topography (poles)",
                       "Orbit x atmospheric transmission 0.75 (Appelbaum & Flood 1990)"),
    "sunlit_frac": ("Astronomical day length x (1 - daytime cloud cover), NASA POWER (CERES SYN1deg)",
                    "0.5 (equator); NASA PGDA average illumination map (poles)", "Astronomical day length"),
    "sun_elevation_deg": ("Astronomy: latitude and axial tilt (23.44 deg)", "Astronomy: axial tilt 1.54 deg",
                          "Astronomy: axial tilt 25.19 deg"),
    "bare_frac": ("ESA WorldCover 10 m; SCAR Antarctic rock outcrop map south of 60 S",
                  f"{_BY_PHYSICS} (regolith and rock)", f"{_BY_PHYSICS} (regolith and rock)"),
}

VARIABLE_NAMES: tuple[str, ...] = tuple(v.name for v in VARIABLES)
VARIABLE_BY_NAME: dict[str, Variable] = {v.name: v for v in VARIABLES}
VAR_DIMENSIONS: tuple[str, ...] = tuple(v.dimension for v in VARIABLES)
DIMENSIONS: tuple[str, ...] = tuple(dict.fromkeys(VAR_DIMENSIONS))

# Quantile levels describing the distribution of a variable inside a tile.
# 0.05 ... 0.95 in steps of 0.05; the median (0.5) is part of the grid.
QUANTILE_LEVELS: np.ndarray = np.round(np.linspace(0.05, 0.95, 19), 2)

# Similarity penalty: quadratic in the gap up to this many tolerances, linear
# beyond (Huber 1964, robust estimation), so that one very poor variable does not decide alone the order of
# the sites. 2.5 lies in the range 2 to 3 that was considered; hypothesis to calibrate (with 2 or 3, a few
# site lists change and the credibility verdicts stay the same).
HUBER_DELTA: float | None = 2.5   # None = purely quadratic (Gaussian) penalty

# --------------------------------------------------------------------- missions
LEVEL_VALUES: dict[str, float] = {"strong": 1.0, "medium": 0.5, "low": 0.2, "none": 0.0}

# Qualitative importance of each dimension per use case.
# Values are hypotheses to justify with references.
MISSION_PROFILES: dict[str, dict[str, str]] = {
    "Rover Mobility": {"Topography": "strong", "Thermal": "medium", "Water / Ice": "none",
                       "Illumination": "medium", "Surface cover": "strong"},
    "Solar Panels Deployment": {"Topography": "medium", "Thermal": "strong", "Water / Ice": "none",
                                "Illumination": "strong", "Surface cover": "medium"},
    "Drilling & In-Situ Resource Utilization (ISRU)": {
        "Topography": "low", "Thermal": "medium", "Water / Ice": "strong", "Illumination": "low",
        "Surface cover": "medium"},
    "Habitat / Construction": {"Topography": "medium", "Thermal": "strong", "Water / Ice": "medium",
                               "Illumination": "medium", "Surface cover": "medium"},
}
# Two targets are flagged as near-duplicates when their best 1 % of Earth's land (ROBUSTNESS_TOP_FRAC) overlap
# at least this much (intersection over union of the land area): their best analog sites are then mostly the
# same places, and they are not two independent results. Display threshold only, hypothesis to calibrate.
TWIN_TARGET_OVERLAP = 0.5

# Why each profile weighs the dimensions this way. Hypotheses, to back with references.
MISSION_RATIONALE: dict[str, str] = {
    "Rover Mobility": "Traversability is driven by slope, roughness and what covers the ground "
                      "(vegetation, snow and ice change wheel-soil contact); thermal swings limit "
                      "batteries and electronics; a solar-powered rover depends on available light.",
    "Solar Panels Deployment": "Energy yield depends on insolation and on the fraction of time in "
                               "sunlight; thermal cycling stresses panels and wiring; the ground "
                               "must be flat and bare enough to anchor the arrays without shading "
                               "or snow; the height of the Sun sets the tilt of the arrays (vertical at the "
                               "lunar poles).",
    "Drilling & In-Situ Resource Utilization (ISRU)": "Water ice is the resource being looked for; "
                                                      "cold temperatures govern ice stability and "
                                                      "drill behaviour in bare rock and soil, not organic "
                                                      "soil or glacier ice; terrain matters less once landed.",
    "Habitat / Construction": "Long stays face the full thermal cycle; flat, bare ground eases building "
                              "on regolith-like foundations; "
                              "local ice and sunlight reduce what must be shipped.",
}

# ----------------------------------------------------------- engine parameters
STATUS_STRONG = 0.8        # similarity >= this  -> "Strong match" (fallback without Earth pairs)
STATUS_PARTIAL = 0.5       # similarity >= this  -> "Partial match", else "Weak match"
# Status from the Earth reference: a site is a strong match on a variable when it is closer to the
# target than this share of random pairs of Earth places are to each other (0.5 = a typical pair).
STATUS_STRONG_CLOSER = 0.75
STATUS_PARTIAL_CLOSER = 0.50
COVERED_THRESHOLD = 0.5    # a dimension is "covered" if the best site reaches this
# A variable is out of Earth's range for a target when even its closest Earth tile is farther
# than this share of the gaps between two random Earth places (type-1 novelty, MESS/ExDet).
REACHABLE_MAX_PCT = 0.99

DEFAULT_RADIUS_KM = 300.0  # non-maximum suppression radius R
NOVELTY_LAMBDA_KM = 500.0  # novelty scale lambda
# Monte-Carlo robustness: each draw picks one concentration level (low =
# weights very uncertain) and, in a third of draws, leaves one variable out (structural uncertainty).
DIRICHLET_ALPHAS = (5.0, 20.0, 50.0)
ROBUSTNESS_DROP_PROB = 1 / 3
# Mission weights strong / medium / low = 1 / 0.5 / 0.2 normalise to ~0.59 / 0.29 / 0.12 for three ranked
# criteria, close to rank-order centroid weights 0.61 / 0.28 / 0.11 (Barron & Barrett 1996).
ROBUSTNESS_TOP_FRAC = 0.01
ROBUSTNESS_DRAWS = 300
SCORE_INTERVAL = (0.05, 0.95)  # quantiles of the perturbed score shown as an interval
POOL_OVERALL = 150         # best distinct sites on the overall score
POOL_PER_DIMENSION = 50    # best distinct sites per weighted dimension ("specialists")
TOP_SITES_SHOWN = 10
CREDIBILITY_MAX_KM = 150.0  # a known analog is matched to the nearest tile within this (a tile is ~110 km)

# Under-studied sites (the challenge asks to look beyond well-known analogs): far from every
# catalogued analog and outside the famous regions the challenge text names.
UNDERSTUDIED_MIN_KM = 500.0
FAMOUS_REGIONS: dict[str, tuple[float, float, float, float]] = {   # lat_min, lat_max, lon_min, lon_max
    "Antarctica": (-90.0, -60.0, -180.0, 180.0),
    "Atacama": (-30.0, -18.0, -72.0, -66.0),
    "Iceland": (63.0, 67.0, -25.0, -13.0),
    "Hawaii": (18.0, 23.0, -161.0, -154.0),
}

# Credibility check. Protocol fixed BEFORE looking at the results:
# the known analogs relevant to the target body must have a mean percentile rank better than
# random places on land (p < alpha over n_draws random sets, score without novelty). With fewer than
# min_sites sites the check is inconclusive. top_frac is the former criterion, still reported.
# Revision of 14 Nov 2026, after the results: ranks are shares of Earth's land area and tied
# tiles share their average rank; both verdicts are kept by the team.
CREDIBILITY_TEST: dict[str, float] = {"top_frac": 0.10, "alpha": 0.05, "n_draws": 10_000, "min_sites": 5}
# What the check can and cannot show, stated next to every verdict. Measured on 14 Nov 2026: the cross-check
# runs the wrong way (Moon analogs rank better for Jezero than for Shackleton, Mars analogs better for Shackleton
# than for Arcadia), and after a Holm correction for the targets only Jezero and Gale pass.
CREDIBILITY_CAVEAT = ("Aridity, ground below 0 C and bare ground are set to 1 by physics for every target, so this "
                      "check mainly shows that dry, cold, bare and rough places rank above random land; it does not "
                      "show that the ranking tells one target from another (see the cross-check with the other "
                      "body). The p-value is also given after a Holm correction for testing every target.")
SMALL_ANALOG_KM = 11.0     # a known analog smaller than a zoom sub-area is diluted in its ~110 km tile

# ------------------------------------------------------------------ knowledge base
# Structural limits of Earth as an analog. Always displayed.
STRUCTURAL_LIMITS: list[dict[str, str]] = [
    {"Factor": "Gravity", "Moon": "About 1/6 g", "Mars": "About 0.38 g",
     "Impact": "Mobility, drilling and fluid tests are not representative."},
    {"Factor": "Atmosphere", "Moon": "Near vacuum", "Mars": "Very thin (about 0.6 kPa)",
     "Impact": "Heat transfer, wind and dust behave differently."},
    {"Factor": "Radiation", "Moon": "No atmosphere, no global field",
     "Mars": "Thin atmosphere, no global field",
     "Impact": "Surface dose is higher than on Earth."},
    {"Factor": "Regolith / dust", "Moon": "Abrasive, electrostatically charged",
     "Mars": "Fine, specific composition",
     "Impact": "Simulants are needed; the analogy is partial."},
]

# What a site can be used for, per dimension, when the analogy holds on it.
TEST_ACTIVITIES: dict[str, tuple[str, ...]] = {
    "Topography": ("Rover traverse, slope climbing and obstacle-avoidance trials",
                   "Landing-site hazard mapping (lidar, stereo imaging)"),
    "Thermal": ("Thermal cycling and cold-soak tests of hardware",
                "Battery and heater sizing for the day/night swing"),
    "Water / Ice": ("Ice prospecting (ground-penetrating radar, neutron spectrometer)",
                    "Drilling and sample handling in dry or frozen ground"),
    "Illumination": ("Solar array sizing and power-budget validation",
                     "Navigation cameras and imaging under comparable lighting"),
    "Surface cover": ("Wheel-soil and anchoring trials on bare, unvegetated ground",
                      "Sampling and instrument tests on exposed rock and soil"),
}
# Ice-prospecting tests only make sense for a target where ice near the surface is documented: ice_proxy is a
# temperature condition, met by every target, ice or not. (documented, source) per target.
ICE_PROSPECTING_ACTIVITY = TEST_ACTIVITIES["Water / Ice"][0]
_LUNAR_POLAR_ICE = (True, "Water ice exposed at the surface of permanently shadowed regions of the lunar south pole "
                          "(Li et al. 2018, PNAS, Chandrayaan-1 M3 spectra).")
TARGET_NEAR_SURFACE_ICE: dict[str, tuple[bool, str]] = {
    "moon_shackleton": _LUNAR_POLAR_ICE,
    "moon_malapert": _LUNAR_POLAR_ICE,
    "moon_mare_tranq_pit": (False, "Equatorial mare: ice is stable only in polar cold traps (Vasavada et al. 1999, "
                                   "Icarus)."),
    "mars_jezero": (False, "Low latitude: ground ice is not stable near the surface (Mellon & Jakosky 1995, JGR); "
                           "water is bound in hydrated minerals."),
    "mars_gale": (False, "Near the equator: ground ice is not stable near the surface (Mellon & Jakosky 1995, JGR); "
                         "Curiosity found water bound in hydrated minerals."),
    "mars_arcadia": (True, "Excess ice tens of metres thick under a thin cover (Bramson et al. 2015, GRL, SHARAD "
                           "radar and terraced craters)."),
    "mars_oxia": (False, "Low latitude: ground ice is not stable near the surface (Mellon & Jakosky 1995, JGR); "
                         "water is bound in clay minerals (Quantin-Nataf et al. 2021, Astrobiology)."),
}

# --------------------------------------------------------- real target profiles
# Target values that are not read from a raster: set by physics or taken from published
# measurements, as a (low, high) range spread uniformly, with their source.
BODY_REFERENCE_VALUES: dict[str, dict[str, tuple[float, float, str]]] = {
    "Moon": {
        "aridity": (1.0, 1.0, "Physics: no liquid water can exist at the surface (vacuum)."),
        "ice_proxy": (1.0, 1.0, "Physics: subsurface temperatures stay far below 273 K all year "
                                "(Diviner, Vasavada et al. 2012)."),
        "bare_frac": (1.0, 1.0, "Physics: no vegetation and no snow cover; the surface is regolith and rock "
                                "(ice in polar cold traps is buried or a trace frost, not a ground cover)."),
    },
    "Mars": {
        "aridity": (1.0, 1.0, "Physics: no stable liquid water at the surface (pressure near the triple point)."),
        "ice_proxy": (1.0, 1.0, "Physics: mean annual ground temperature below 240 K at all latitudes."),
        "bare_frac": (1.0, 1.0, "Physics: no vegetation; regolith and rock at the surface of the targets "
                                "(thin seasonal frost is not a permanent cover, as in Earth land-cover maps)."),
    },
}
# Published in-situ ranges of the Mars thermal variables. The target profile comes from the
# Mars Climate Database; these ranges only control it (shown in the provenance of each value).
_VIKING2 = "Viking Lander 2 at 48 N as latitude analogue; CO2 frost point ~148 K, approximate."
TARGET_REFERENCE_VALUES: dict[str, dict[str, tuple[float, float, str]]] = {
    "mars_gale": {
        "diurnal_amp_k": (55.0, 95.0, "Curiosity REMS ground temperature (Martinez et al. 2017), approximate."),
        "temp_min_k": (178.0, 205.0, "Curiosity REMS ground temperature (Martinez et al. 2017), approximate."),
    },
    "mars_jezero": {
        "diurnal_amp_k": (65.0, 100.0, "Perseverance MEDA ground temperature (2021-2023), approximate."),
        "temp_min_k": (175.0, 200.0, "Perseverance MEDA ground temperature (2021-2023), approximate."),
    },
    "mars_arcadia": {
        "diurnal_amp_k": (35.0, 70.0, _VIKING2),
        "temp_min_k": (148.0, 195.0, _VIKING2),
    },
}
SOLAR_CONSTANT_WM2 = 1361.0
MARS_ATMOSPHERE_TRANSMISSION = 0.75  # global irradiance / top of atmosphere for a dust opacity ~0.5
#                                     (Appelbaum & Flood 1990, NASA TM-102299)

# ------------------------------------------------------------- known analog sites
@dataclass(frozen=True)
class KnownAnalog:
    """A documented terrestrial analog site, with two sources (coordinates; analog role).

    mars / moon: the properties the site is a documented analog for, used by the credibility
    check. Each item is a variable name (config.VARIABLE_NAMES) or, when
    the documented role covers all of it, a dimension name. Encode the exact property the source
    names (e.g. Atacama = hyperaridity -> "aridity"), never a broader dimension: a dimension mixes
    variables, and a site famous for one of them would be judged on the others too.
    () = not tested for that body (e.g. famous for chemistry, which we do not model).
    Every site counts for the novelty axis (4.3).
    """

    name: str
    lat: float
    lon: float
    bodies: str
    mars: tuple[str, ...]
    moon: tuple[str, ...]
    extent_km: float
    source: str


_W = "Coordinates: Wikipedia"
_WTAS = "Coordinates: Wikipedia 'Terrestrial analogue site' table"
_PD14 = "Preston & Dartnell 2014 (Int. J. Astrobiology)"
_ARTEMIS = "NASA/USGS Artemis volcanic field sites (NTRS 20220013521)"
_PANGAEA = "ESA PANGAEA training"
TO = "Topography"                                          # crater / volcanic terrain: slope and roughness
ARID, FROZEN, COLD = "aridity", "ice_proxy", "temp_min_k"  # exact documented properties
# Catalogue frozen on 2026-11-13, before looking at credibility results (protocol fixed in advance).
# Revision on 2026-11-13, after a first run: the Mars sites were coded with whole dimensions
# ("Water / Ice" = aridity + frozen ground), so hot deserts famous for aridity alone were judged on
# frozen ground too. Each site is now coded with the exact property its sources name. Sites and
# sources are unchanged; the first-run results are kept by the team.
KNOWN_ANALOGS: tuple[KnownAnalog, ...] = (
    # --- Mars analogs, tested on the exact properties they are documented for
    KnownAnalog("Atacama Desert (hyperarid core)", -24.5, -69.25, "Mars", (ARID,), (), 300,
                f"{_WTAS}. Role (hyperaridity): Azua-Bustos et al. 2022; {_PD14}."),
    KnownAnalog("McMurdo Dry Valleys", -77.4667, 162.5167, "Moon;Mars", (ARID, FROZEN, COLD), (), 70,
                f"{_W}. Role (cold hyperarid polar desert, permafrost): Marchant & Head 2007; {_PD14}."),
    KnownAnalog("Beacon Valley (Antarctica)", -77.8167, 160.65, "Mars", (ARID, FROZEN, COLD), (), 15,
                f"{_W}. Role (cold desert, buried ice): {_PD14}."),
    KnownAnalog("Haughton Crater (Devon Island)", 75.3833, -89.6667, "Moon;Mars", (FROZEN, COLD), (TO,), 23,
                f"{_WTAS}. Role (polar desert, ground ice; impact crater): {_PD14}; CMU/GSA Haughton lunar study."),
    KnownAnalog("Axel Heiberg Island", 79.75, -91.0, "Mars", (FROZEN,), (), 150,
                f"{_W}. Role (permafrost, cold springs): {_PD14}."),
    KnownAnalog("Bockfjorden, Svalbard (AMASE)", 79.4317, 13.305, "Moon;Mars", (FROZEN,), (), 30,
                f"{_W} (Sverrefjellet). Role (permafrost, rock glaciers): {_PD14}; Wikipedia analogue table (AMASE)."),
    KnownAnalog("Qaidam Basin", 37.2667, 94.45, "Mars", (ARID, TO), (), 250,
                f"{_W}. Role (hyperarid, yardangs): {_PD14}; Europlanet TA1; Meng et al. 2025 (arXiv 2501.08584)."),
    KnownAnalog("Mojave Desert", 35.0, -116.0, "Mars", (ARID, TO), (), 200,
                f"{_W}. Role (arid desert, rover testing): {_PD14}; GSA Special Paper 483."),
    # --- Moon analogs (Artemis / PANGAEA training sites), tested on topography
    KnownAnalog("Meteor Crater (Arizona)", 35.0281, -111.0233, "Moon;Mars", (), (TO,), 1.2,
                f"{_W}. Role: IAC 2021 Artemis geology training; Foucher et al. 2021."),
    KnownAnalog("SP Crater (San Francisco volcanic field)", 35.5822, -111.6318, "Moon", (), (TO,), 2,
                f"{_W}. Role: {_ARTEMIS}."),
    KnownAnalog("Kilbourne Hole (Potrillo volcanic field)", 31.9719, -106.9647, "Moon", (), (TO,), 3,
                f"{_W}. Role: {_ARTEMIS}."),
    KnownAnalog("Mistastin crater (Labrador)", 55.8833, -63.3, "Moon", (), (TO,), 28,
                f"{_W}. Role: IAC 2021 Artemis capstone training; SpaceQ 2023."),
    KnownAnalog("Nördlinger Ries", 48.8833, 10.5667, "Moon", (), (TO,), 24, f"{_W}. Role: {_PANGAEA}."),
    KnownAnalog("Askja caldera (Iceland)", 65.033, -16.783, "Moon", (), (TO,), 50,
                f"{_W}. Role: NASA Artemis Iceland training (NTRS 20220013521)."),
    KnownAnalog("Timanfaya (Lanzarote)", 28.9942, -13.7933, "Moon;Mars", (), (TO,), 15, f"{_W}. Role: {_PANGAEA}."),
    # --- famous analogs for properties we do not model: novelty axis only
    KnownAnalog("Rio Tinto", 37.404, -6.3358, "Mars", (), (), 20, f"{_WTAS}. Role: {_PD14}; Foucher et al. 2021."),
    KnownAnalog("Mauna Kea (Hawaii)", 19.8206, -155.4681, "Moon;Mars", (), (), 40,
                f"{_W}. Role: Wikipedia analogue table."),
    KnownAnalog("Kilauea (Hawaii)", 19.425, -155.2919, "Moon;Mars", (), (), 20,
                f"{_WTAS}. Role: Wikipedia analogue table."),
    KnownAnalog("Mars Desert Research Station (Utah)", 38.4065, -110.7919, "Moon;Mars", (), (), 20,
                f"{_W}. Role: USGS Utah analogs."),
    KnownAnalog("Death Valley", 36.2469, -116.8169, "Mars", (), (), 100, f"{_W}. Role: weakly sourced."),
    KnownAnalog("Danakil Depression", 14.2417, 40.3, "Mars", (), (), 100, f"{_W}. Role: Europlanet TA1."),
    KnownAnalog("Concordia Station (Antarctica)", -75.1, 123.33, "Moon;Mars", (), (), 50,
                f"{_WTAS}. Role: human-factors analog (ESA)."),
    KnownAnalog("CHILL-ICE lava tube (Iceland)", 64.7869, -20.7144, "Moon;Mars", (), (), 5, f"{_WTAS}. Role: same."),
    KnownAnalog("Mount Etna", 37.755, 14.995, "Moon;Mars", (), (), 40, f"{_W}. Role: Foucher et al. 2021."),
    KnownAnalog("Teide (Tenerife)", 28.2731, -16.6394, "Moon;Mars", (), (), 20, f"{_W}. Role: Foucher et al. 2021."),
    KnownAnalog("Sudbury Basin", 46.6, -81.1833, "Moon;Mars", (), (), 60, f"{_WTAS}. Role: same."),
    KnownAnalog("Kangerlussuaq (Greenland)", 67.0086, -50.6892, "Mars", (), (), 50, f"{_W}. Role: Europlanet TA1."),
    # --- added on 2026-11-14, after the results: documented analogs found near sites the app called
    # under-studied. Novelty axis only (no property tested), so the credibility protocol is unchanged.
    KnownAnalog("Ladakh (Tso Kar, Puga)", 33.3, 77.983, "Mars", (), (), 50,
                f"{_W} (Tso Kar). Role (high-altitude cold desert, brines, permafrost): Pandey et al. 2020, Int. J. "
                "Astrobiology 19(1), 78-98."),
    KnownAnalog("Lake Salda (Turkey)", 37.55, 29.683, "Mars", (), (), 10,
                f"{_W}. Role (carbonates and delta deposits like Jezero's): NASA Earth Observatory, 17 Feb 2021; "
                "Horgan et al. (Purdue) field work 2019."),
)

# ------------------------------------------------------- subsurface case studies
# Case-study layer: caves have no global raster, so this is a sourced
# catalogue shown next to the ranking and never part of the score.
@dataclass(frozen=True)
class SubsurfaceFeature:
    """A documented pit or cave of a planetary target."""

    name: str
    geometry: str
    temperature_k: float | None   # interior temperature, None when it is not documented
    temperature_note: str
    source: str


SUBSURFACE_TARGETS: dict[str, SubsurfaceFeature] = {
    "moon_mare_tranq_pit": SubsurfaceFeature(
        "Mare Tranquillitatis pit",
        "Collapse pit that may open onto a lava tube: inner opening 100 x 88 m, 105 m deep, funnel 146 x 140 m.",
        290.0,
        "Beyond the opening, in permanent shadow, the temperature stays near 290 K (17 C) and varies by less "
        "than 1 K over a lunar day: a thermophysical model constrained by LRO Diviner measurements, not a "
        "direct measurement.",
        "Geometry: LROC Lunar Pit Atlas (NASA/GSFC/Arizona State University). Temperature: Horvath et al. 2022, "
        "Geophysical Research Letters, doi:10.1029/2022GL099710. Lunar pits were first seen by JAXA's SELENE "
        "(Kaguya) in 2009."),
}
# Why the surface climate says something about a cave. Empirical rule of cave climatology (e.g. Badino 2010,
# Acta Carsologica 39(3); Sebela & Turk 2011, Theor. Appl. Climatol.); hypothesis to calibrate per cave.
CAVE_TEMPERATURE_RULE = ("Away from its entrance, a cave on Earth stays close to the mean annual air temperature "
                         "of the surface above it; cold-air traps (ice caves) and strongly ventilated or shallow "
                         "tubes depart from it.")
# Earth lava-tube fields: (name, country, lat, lon). Coordinates: Wikipedia article of the cave, the park or
# the volcanic field (read on 14 Nov 2026), so they locate the field, not a cave entrance.
LAVA_TUBE_SOURCE = "Coordinates: Wikipedia (cave, park or volcanic field article), read on 14 Nov 2026"
LAVA_TUBE_SITES: tuple[tuple[str, str, float, float], ...] = (
    ("Cueva de los Verdes (La Corona tube, Lanzarote)", "Spain", 29.1611, -13.4389),
    ("Cueva del Viento (Tenerife)", "Spain", 28.3467, -16.7222),
    ("Gruta das Torres (Pico, Azores)", "Portugal", 38.4943, -28.5024),
    ("Surtshellir", "Iceland", 64.7842, -20.7277),
    ("Manjanggul (Jeju)", "South Korea", 33.5282, 126.7701),
    ("Aokigahara lava caves (Mount Fuji)", "Japan", 35.4747, 138.6664),
    ("Krong No volcanic caves (Dak Nong)", "Vietnam", 12.333, 107.833),
    ("Undara lava tubes (Queensland)", "Australia", -18.2011, 144.5961),
    ("Budj Bim lava caves (Victoria)", "Australia", -38.0628, 141.9256),
    ("Rangitoto Island lava caves", "New Zealand", -36.7867, 174.8601),
    ("Harrat Khaybar (Umm Jirsan tube)", "Saudi Arabia", 25.7194, 39.9428),
    ("Mount Suswa lava caves", "Kenya", -1.15, 36.35),
    ("Chyulu Hills (Leviathan Cave)", "Kenya", -2.6, 37.85),
    ("Piton de la Fournaise lava tunnels (Reunion)", "France", -21.2425, 55.7089),
    ("Santa Cruz lava tunnels (Galapagos)", "Ecuador", -0.6255, -90.3679),
    ("Hawaii Volcanoes National Park (Nahuku tube)", "United States", 19.3833, -155.2),
    ("Lava Beds National Monument (California)", "United States", 41.7139, -121.5083),
    ("Lava River Cave (Oregon)", "United States", 43.8954, -121.3689),
    ("Craters of the Moon (Idaho)", "United States", 43.4617, -113.5627),
    ("El Malpais (New Mexico)", "United States", 34.8772, -108.0508),
    ("Pisgah Crater lava tubes (Mojave)", "United States", 34.7464, -116.375),
)

# Dimensions of the official challenge text that this version does not evaluate.
NOT_EVALUATED: tuple[str, ...] = (
    "Geology and mineralogy (only the share of bare ground is compared)",
    "Dust, wind and aeolian processes",
    "Caves and subsurface environments (case studies only, for the Mare Tranquillitatis pit)",
    "Biology and habitability",
)

# How the app answers the official challenge text, item by item (home page): the 4 objectives and the
# 12 "potential considerations" of the Details tab (read on 13 Nov 2026).
# (kind, challenge text, status, our answer, page). The status is a plain statement of what this version
# does, not a self-assessment of quality: keep it honest when a feature is added or removed.
CHALLENGE_STATUSES: tuple[str, ...] = ("Covered", "Partly covered", "Not covered")
CHALLENGE_MAP: tuple[tuple[str, str, str, str, str], ...] = (
    ("Objective", "Use open NASA Earth data together with Moon and Mars data from NASA, other agencies and "
     "open sources", "Covered",
     "NASA POWER (Earth climate), LRO LOLA and Diviner, NASA PGDA illumination, MGS MOLA; ESA Copernicus DEM "
     "(Earth relief), ESA WorldCover and the SCAR Antarctic rock outcrop map (Earth bare ground); Mars surface "
     "temperature from the Mars Climate Database, a climate model by LMD and "
     "partners supported by ESA and CNES.", "2"),
    ("Objective", "Identify and characterize Earth locations that are analogs of candidate Moon-base sites or "
     "future Mars landing and exploration sites", "Covered",
     "Every land tile of Earth is ranked against 3 lunar and 4 Martian targets, among them one of NASA's nine "
     "Artemis III candidate regions (Malapert Massif) and two future sites: Oxia Planum (ESA's Rosalind Franklin "
     "rover) and Arcadia Planitia; distinct sites, a novelty axis and the best under-studied site.", "3"),
    ("Objective", "Explain what makes the analogy strong, where it breaks down and how the site helps "
     "exploration", "Covered",
     "Profile by dimension with statuses, rule-based limits statements and a 'what could be tested here' "
     "site sheet.", "3, 5"),
    ("Objective", "Present the findings (map, visualization, tool)", "Covered",
     "Interactive tool: maps, radars, distribution charts; downloadable site sheet, candidate list and a "
     "written field atlas for the public.", "All"),
    ("Consideration", "Think beyond visual similarity", "Covered",
     "Sites are compared on measured physical quantities (slope, roughness, temperature, aridity, ground "
     "below 0 C at depth, sunlight, height of the Sun, bare ground), never on appearance.", "3"),
    ("Consideration", "Consider different dimensions of analogy", "Partly covered",
     "Evaluated: topography, temperature, illumination, water and ice, surface cover (bare ground against "
     "vegetation, snow and ice), operations (mission profiles). Case studies only: subsurface. Not evaluated: "
     "geology and mineralogy, dust and wind, biology.", "2, 5"),
    ("Consideration", "An analog does not have to match everything: say what is similar, how similar, what "
     "differs", "Covered",
     "Non-compensatory score, similarity per variable, statuses referenced to how different two places on "
     "Earth are.", "3"),
    ("Consideration", "Think about processes as well as places", "Not covered",
     "The engine compares states (distributions over the year), not processes such as dust storms, volcanism "
     "or erosion.", "5"),
    ("Consideration", "Look beyond well-known analog sites", "Covered",
     "Novelty axis against a sourced catalogue of known analogs; the best site outside Antarctica, the "
     "Atacama, Iceland and Hawaii is highlighted.", "3"),
    ("Consideration", "Explore the whole Earth, including below its surface", "Partly covered",
     "Every land tile is screened. Below the surface: a sourced case study, 21 lava-tube fields compared "
     "with the Mare Tranquillitatis pit, not a global search.", "3, 5"),
    ("Consideration", "Use multiple datasets", "Covered",
     "Nine sources brought to a common scale and compared variable by variable.", "2"),
    ("Consideration", "Consider scale: a crater, a valley, a small area", "Partly covered",
     "Screening on 1 x 1 degree tiles (about 110 km), then a zoom on sub-areas of about 11 km inside the best "
     "sites, terrain only: climate data stay at tile scale.", "3"),
    ("Consideration", "Connect your analog to exploration", "Covered",
     "Four mission profiles weigh the dimensions; the site sheet lists representative tests, tests to run "
     "with caution, the months when the Sun stands as on the target, and what never transfers from Earth.",
     "2, 5"),
    ("Consideration", "Be creative with your approach", "Covered",
     "Distances between distributions (Wasserstein), complementary sets of sites, Monte-Carlo robustness and "
     "a credibility test against random land tiles.", "3, 4"),
    ("Consideration", "Remember Earth's limits as an analog", "Covered",
     "Structural limits (gravity, vacuum or thin atmosphere, radiation, regolith) are always shown; variables "
     "no place on Earth can match leave the score and are reported with figures.", "5"),
    ("Consideration", "Surprise us", "Covered",
     "Instead of one winner: a set of 2-3 sites that cover each other's gaps, a quantified list of what Earth "
     "cannot reproduce, and a reverse atlas telling which world any place on Earth resembles.", "4, 5, 6"),
)
