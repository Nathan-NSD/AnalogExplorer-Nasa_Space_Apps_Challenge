# Method

AnalogExplorer compares every land tile of the Earth with seven lunar and Martian targets, dimension by
dimension, and reports what Earth cannot reproduce. This page describes the method; the code that implements it
is in `src/math_engine.py` (pure calculation) and `src/analysis.py` (how the engine is applied to a dataset). All
thresholds below are constants of `src/config.py`.

## 1. Design principles

1. **Characterize before ranking.** A single score hides discrepancies. The main output is a profile by
   dimension; the score is a summary.
2. **Same physics on three worlds.** Each variable is defined physically (for example, slope in degrees over a
   ~463 m step) and computed with the same operator on the Earth, the Moon and Mars.
3. **Compare distributions, not averages.** A place is described by the spread of its conditions, not by one
   mean value.
4. **Report what does not match.** Structural differences (gravity, atmosphere, radiation) and conditions that
   no place on Earth reaches are part of the result.
5. **Heavy work offline, interaction online.** Data are processed once; the app only runs fast matrix
   operations, so every slider responds immediately.

## 2. Targets, dimensions and variables

**Targets (7).** Moon: Shackleton crater region and Malapert massif (south pole, Artemis candidate regions),
Mare Tranquillitatis pit (a non-polar site with a subsurface opening). Mars: Jezero crater (Perseverance),
Gale crater (Curiosity), Arcadia Planitia (ice-rich mid-latitudes, a candidate for future crewed missions) and
Oxia Planum (landing site of ESA's Rosalind Franklin rover). Each target is a window of about 110 km, the size
of an Earth tile.

**Earth tiles (23,039).** Cells of 1° × 1° (about 110 km) with at least 25 % of land.

**Dimensions and variables (5 and 10).**

| Dimension | Variables |
|---|---|
| Topography | slope, roughness |
| Thermal | diurnal temperature amplitude, minimum temperature |
| Water / Ice | aridity index, ground below 0 °C at depth |
| Illumination | mean insolation, sunlit fraction, Sun elevation in daytime |
| Surface cover | bare ground |

Sources and definitions of each variable: [data.md](data.md) §1. Water ice itself (hydrogen, radar signatures)
cannot be measured the same way on three worlds, so the comparison uses frozen ground, defined by the same
physics everywhere.

## 3. From data to ranking

**1. Distributions.** For every tile and every target, each variable is summarized by 19 quantiles (5 % to
95 %). The distribution covers space and the year: the pixels of the tile for topography, the cells × 12 months
(or 12 Martian seasons) for climate and illumination.

**2. Gap per variable.** The one-dimensional Wasserstein distance, i.e. the mean gap between the two quantile
curves:

$$W_{1,k}(t, T) = \frac{1}{19}\sum_{i=1}^{19} \left| q_{k,i}^{(t)} - q_{k,i}^{(T)} \right|$$

**3. Tolerance.** The gap is read in tolerances, $z = W_1 / \sigma_k$, where $\sigma_k$ is the median gap between
two random places on Earth's land (places drawn in proportion to their land area). A site is "similar" when it is
closer to the target than two ordinary places on Earth are to each other. For a bounded variable where many pairs
are identical (bare ground, frozen ground), $\sigma_k$ = P(two places differ) × the median gap of the pairs that
differ.

**4. Reachability.** Before ranking, the best Earth gap of each variable is placed in the distribution of
Earth-Earth gaps. If even the closest tile is farther than 99 % of those gaps, the variable is **out of Earth's
range**: it leaves the score and is reported with figures in the limits report (the logic of novelty detection
in species-distribution models, such as MESS and ExDet). Without this step, a condition no place on Earth reaches
(the lunar day-night amplitude) would decide the whole ranking.

**5. Similarity.** $\ln s = -z^2/2$ up to $z = 2.5$, then linear, $\ln s = -(2.5\,z - 2.5^2/2)$ (Huber penalty):
a very distant variable still costs, but does not dictate the order alone.

**6. Mission weights.** Four mission profiles (rover mobility, solar panel deployment, drilling and in-situ
resource utilization, habitat and construction) set the importance of each dimension: strong = 1, medium = 0.5,
low = 0.2, none = 0. Each dimension weight is spread equally over its variables, then $\sum_k w_k = 1$. An expert
mode lets the user set the weights directly.

| Mission | Topography | Thermal | Water / Ice | Illumination | Surface cover |
|---|---|---|---|---|---|
| Rover mobility | strong | medium | none | medium | strong |
| Solar panels | medium | strong | none | strong | medium |
| Drilling / ISRU | low | medium | strong | low | medium |
| Habitat / construction | medium | strong | medium | medium | medium |

**7. Score.** A weighted geometric mean, which is non-compensatory: an excellent variable cannot hide a very poor
one.

$$S(t) = \prod_k s_{tk}^{\,w_k} = \exp\Big(\sum_k w_k \ln s_{tk}\Big)$$

A Mahalanobis or cosine distance was not used: the first is invariant to rescaling (weights would have no
effect), the second compares directions, not magnitudes.

**8. Distinct sites.** Neighbouring tiles are alike. Non-maximum suppression keeps the best tile, removes those
within 300 km (adjustable), and repeats: the result is a list of distinct sites, not blocks of adjacent cells.

**9. Novelty.** The challenge asks to look beyond well-known analogs. Novelty is shown on a separate axis:
$N = 1 - e^{-d/500\,\mathrm{km}}$, with $d$ the distance to the nearest of 29 catalogued analogs (sourced).
A slider moves the ranking along the similarity-novelty trade-off; non-dominated sites (Pareto front) are
highlighted. A site is called under-studied when it lies more than 500 km from any catalogued analog and outside
the famous analog regions (Antarctica, the Atacama, Iceland, Hawaii).

## 4. Reading a result

The raw score is never shown alone. Each site comes with:
- its **relative score** (1 = best Earth tile for this target and mission);
- **closer than X % of Earth pairs**: the share of random pairs of places on Earth that are less alike than the
  site and the target (50 % = as alike as two ordinary places on Earth);
- **top X % of Earth's land**: the share of land area that scores at least as well.

Every share of the Earth is a share of land area, not a count of tiles, so the many narrow tiles near the poles
do not weigh more than their surface. A variable or dimension is a **strong match** when the site is closer to
the target than 75 % of Earth pairs, a **partial match** above 50 %, a **weak match** otherwise.

## 5. Robustness

Mission weights are expert judgements. The robustness of a site is its probability of staying in the best 1 % of
Earth's land when the weights are perturbed: 300 Monte-Carlo draws from a Dirichlet distribution centred on the
mission weights, with a concentration drawn among 5, 20 and 50 (strong to weak uncertainty), and one variable
left out in a third of the draws (structural uncertainty, as recommended for composite indicators by the
OECD/JRC handbook). The same draws give a 90 % range of the relative score. The qualitative weights 1 / 0.5 /
0.2 are close to rank-order centroid weights (Barron & Barrett 1996).

## 6. Complementary ensemble

No single place reproduces another world, but different places can match different conditions. The app looks
for the set $S$ of $k \le 3$ sites that best covers the target:

$$C(S) = \sum_k w_k \max_{j \in S} s_{jk}$$

This coverage function is monotone and submodular; for $k \le 3$ the best set is found exactly among the
candidates (the best distinct sites plus the best site of each weighted dimension, which may score poorly
overall but excel where the others fail). A variable whose best similarity in the set stays below 0.5 is
reported as **not covered**. The app shows the gain of complementarity over the best single site.

## 7. Fine-scale zoom

The challenge asks to consider scale: an analog can be a crater or a valley, not a 110 km cell. For the sites
shown, each tile is cut into sub-areas of about 11 km; slope and roughness are recomputed there with the same
elevation data, pixel and operators, and compared with the same target profile. Climate stays at the scale of
the tile (its grids are 0.5° to 1°), which the app states.

## 8. Credibility check

The goal is not to rediscover known analogs, but they test whether the method is sound. The protocol was fixed
before looking at the results:
- each known analog of the sourced catalogue is ranked among all tiles on the exact properties its sources
  document (the Atacama on aridity, permafrost sites on frozen ground, Artemis training craters on topography);
- the mean percentile rank of the known sites is compared with 10,000 random sets of places on land;
- the check passes if p < 0.05; with fewer than 5 sites it is inconclusive.

The app also reports a Holm correction for testing seven targets, a cross-check (the same sites ranked for a
target of the other body), the analogs smaller than a zoom sub-area, and the fact that the variables fixed by
physics are the same for every target. Results are reported as they are, including failures: the check fails for
the three lunar targets and passes for the four Martian targets. The protocol and its revisions are stated in the
app (page 3, Credibility check).

## 9. Limits report

For each target and site, the app lists:
- **structural limits**: gravity (1/6 g on the Moon, 0.38 g on Mars), vacuum or a thin atmosphere (about
  0.6 kPa on Mars), radiation without a global magnetic field, abrasive regolith;
- **variables out of Earth's range**, with the target value, the closest Earth tile and the gap;
- **dimensions not evaluated**: geology and mineralogy (only bare ground is compared), dust and wind, biology;
- **where the analogy holds and where it breaks**, variable by variable, generated by rules (no language model);
- a **site sheet**: what could be tested there, what transfers and what does not, and the months when the
  lighting comes closest to the target's ("When to go").

Subsurface conditions are treated as case studies outside the score: 21 lava tubes on Earth are compared with
the thermal environment modelled inside the Mare Tranquillitatis pit. The Earth Atlas page answers the reverse
question: which world does a given place on Earth resemble?

## References

- Barron, F. H. & Barrett, B. E. 1996 — rank-order centroid weights.
- Foucher, F. et al. 2021 — functional analogs in planetary exploration.
- Holm, S. 1979 — sequentially rejective multiple test procedure.
- Huber, P. J. 1964 — robust estimation, Huber loss.
- Mazarico, E. et al. 2011 — illumination conditions of the lunar polar regions (LOLA).
- Nemhauser, G. L., Wolsey, L. A. & Fisher, M. L. 1978 — approximations for maximizing submodular functions.
- OECD / JRC 2008 — Handbook on Constructing Composite Indicators.
- Preston, L. J. & Dartnell, L. R. 2014 — planetary habitability and terrestrial analogues.
- Williams, J. W. et al. 2007 — novel and disappearing climates (no-analog conditions).

Data references (LOLA, Diviner, MOLA, WorldCover, SCAR, Mars Climate Database, Natural Earth): [data.md](data.md) §2.
