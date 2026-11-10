# Urban UFLUX analysis

Analysis code accompanying the manuscript **“Disproportionate tree carbon uptake and sustainable urban greening beyond green space area”** by Songyan Zhu, Wenquan Dong, Jian Xu, Mana Gharun, Zhonghua Zheng, Lanhui Wang, Man Chen and Jadu Dash.

This repository contains an analysis-only Python script extracted from the research notebooks. The manuscript is being prepared for submission to *Nature Cities*; this description does not imply publication or acceptance.

## About the study

The study examines urban ecosystem carbon and water functioning across **72 cities during 2016–2025**, using 30-m UFLUX estimates together with land-cover, meteorological, population and nighttime-light data. It investigates how vegetation composition, particularly tree cover, relates to carbon uptake and water use, and how land-cover transitions and location within cities relate to changes in net ecosystem production (NEP).

The manuscript reports that trees contribute disproportionately to estimated positive NEP relative to their mapped area. It also examines carbon gained per unit of evapotranspiration and the water-balance implications of evapotranspiration exceeding local precipitation in dry cities. Land-cover transition analyses and centre–periphery spatial comparisons provide additional context for assessing urban greening beyond green-space extent.

NEP represents ecosystem carbon uptake after ecosystem respiration. It is not a complete municipal carbon balance incorporating fossil-fuel emissions. Positive-NEP contributions and signed city-wide NEP are distinct quantities.

## Code overview

[`urban_uflux_analysis.py`](urban_uflux_analysis.py) contains the extracted analysis workflows. Source cell numbers are recorded in the script, using zero-based notebook indexing.

The primary source is `Urban_FLUX_v3.ipynb`, with `Urban_FLUX_analysis.ipynb` used to compare shared functions and earlier analysis workflows. The original notebooks and manuscript files are not required to launch the script and are not included in this code release.

| Stage | Analysis | Main output location under the data root |
| --- | --- | --- |
| `summary` | Period means, land-cover composition, observed and adjusted changes, legacy Moran/LISA statistics, and exploratory driver regressions | `4_analysis/summary/<city>/` |
| `tables` | Cross-city NEP, land-cover and transition summaries from existing summary CSV files | `5_supp/cross_city_tables/` |
| `drivers` | Ridge regression, variance inflation factors, and cross-city model performance and importance summaries | `5_supp/analysis_exports/` |
| `tree` | Area-weighted tree-cover shares and contributions to positive and signed NEP | `5_supp/tree_audit/tree_baseline_fast.csv` |
| `robustness` | Flux consistency diagnostics, annual city fluxes and sensitivity scenarios | `5_supp/tree_audit/` |
| `ablation` | Comparison of apparent GPP-dependent efficiency ratios with exogenous efficiency predictors | `5_supp/exogenous_ablation/` |
| `spatial` | Area-normalised centre–periphery LISA comparisons for baseline NEP and its change | `4_analysis/extra_lisa_200m_99perm/` |
| `socioeconomic` | Population-normalised NEP and nighttime-light co-trend analysis | `4_analysis/UrbanEcoCorr/` |

The extraction excludes Earth Engine downloads, satellite retrieval, UFLUX flux-model training and upscaling, GenAI calls, obsolete pilot analyses and duplicate plotting cells. Statistical driver regressions are retained as part of the analysis.

## Installation

Use **Python 3.10 or later** in a separate environment. Install the scientific dependencies:

```bash
python -m pip install numpy pandas scipy xarray netCDF4 geopandas rasterio rioxarray pyproj scikit-learn xgboost statsmodels joblib tqdm
```

The `summary` and `spatial` stages additionally require:

```bash
python -m pip install libpysal esda
```

The `summary`, `spatial` and `socioeconomic` stages use the notebooks’ `geoxai` dependency for raster loading and meteorological utilities:

```bash
python -m pip install git+https://github.com/songyanzhu/geoxai.git
```

Original package versions were not recorded in the supplied notebooks. These commands describe dependencies rather than a validated, version-pinned reproduction environment. The script does not install packages or download datasets automatically.

## Input data

Set `--data-root` to the local `urban_sust` data directory. Preserve the original directory layout:

```text
urban_sust/
├── 2_geojsons/
│   └── <city>.geojson
├── 4_analysis/
│   ├── spatial_EVI_DynamicWorld_UFLUXv1/
│   │   └── <city>_uflux.nc
│   └── summary/
│       └── <city>/
│           └── ... existing summary CSV and joblib files
├── 1_griddata/
│   └── <city>/
│       ├── <city>_ERA5_01deg/
│       ├── <city>_Aqua_temperature_1km/
│       ├── <city>_Terra_temperature_1km/
│       └── <city>_VIIRS_Nightlight_500m/
└── 1_Worldpop/
    └── <ISO3>_<product>_<year>.tif
```

For city raster inputs, the original fallback directory `1_griddata_202605/<city>/` is also supported.

Annual UFLUX NetCDF files must have `time`, `latitude` and `longitude` coordinates. Tree and robustness workflows expect one annual grid for each year from 2016 through 2025. Required variables depend on the stage:

- `tree`: `NEP` and `LC`.
- `robustness`: `NEP`, `LC`, `GPP`, `RECO` and `EVI`; `NEP_dir` is used when available.
- `summary`, `spatial` and `socioeconomic`: the existing UFLUX analysis grids and ERA5 inputs expected by `load_city_uflux()`.
- `tables`: existing city summary CSV files, or outputs from `summary`.
- `drivers` and `ablation`: existing `UFLUX_multiyear_lucc_ML_casual.joblib` files, or outputs from `summary`.
- `socioeconomic`: WorldPop rasters and city-level Aqua, Terra and VIIRS rasters in addition to the UFLUX and ERA5 inputs.

Flux units follow the source notebooks: NEP, GPP, RECO and NEP_dir are in g C m⁻² d⁻¹; ET is in mm yr⁻¹; Dynamic World land-cover codes range from 0 to 8. The required ERA5 band ordering and unit conversions are specified in `load_era5()`.

The manuscript’s Data Availability statement identifies the study’s annual flux grids and city summary tables at [Zenodo, DOI: 10.5281/zenodo.24105442](https://doi.org/10.5281/zenodo.24105442). This reference is taken from the supplied manuscript; archive contents and compatibility with every script stage have not been independently verified. Supporting raster inputs may need to be obtained separately. No research data are bundled with this script.

## Running the analyses

Display the embedded documentation:

```bash
python urban_uflux_analysis.py --help
```

Run a selected workflow against your local data:

```bash
python urban_uflux_analysis.py --data-root "/path/to/urban_sust" --stage tree
python urban_uflux_analysis.py --data-root "/path/to/urban_sust" --stage robustness
```

On Windows, a data path can be supplied as follows:

```powershell
python urban_uflux_analysis.py --data-root "D:\data\urban_sust" --stage tree
```

To generate city summaries and then aggregate or analyse them:

```bash
python urban_uflux_analysis.py --data-root "/path/to/urban_sust" --stage summary
python urban_uflux_analysis.py --data-root "/path/to/urban_sust" --stage tables
python urban_uflux_analysis.py --data-root "/path/to/urban_sust" --stage drivers
python urban_uflux_analysis.py --data-root "/path/to/urban_sust" --stage ablation
```

The `spatial` and `socioeconomic` stages run separately using the same command pattern. There is no automatic “run all” command. Outputs are written under the data root, and several original workflows skip existing outputs. Use a separate clean data directory when a complete recalculation is required. Load joblib files only from trusted sources.

## Reproducibility and interpretation

This release preserves the source analysis formulas and parameter choices. It adds configurable paths, explicit imports, command-line entry points, CSV exports and selected runtime guards. NumPy is seeded with 42 at the command-line entry point; this does not guarantee identical results across all dependencies or stochastic spatial routines.

The following distinctions matter when interpreting outputs:

- The original Ridge routine reports **in-sample** R² and RMSE. Cross-validation selects its regularisation strength; the reported metrics are not held-out performance. The original ordinary linear regression also scales predictors before splitting the data.
- The ablation retains the source’s zero filling, independently sampled subsets and within-sample feature importance. Its theoretical LUE/IWUE formulation and units require author verification before scientific interpretation.
- Legacy spatial statistics in `summary` use the full-period mean and angular radial distances. The separate `spatial` workflow uses the 2016–2020 baseline and metric coordinates, retaining the source defaults of approximately 200 m and 99 permutations. These outputs should not be treated as interchangeable or labelled as 999-permutation results.
- Workflow-specific city exclusions and masks are retained. The study domain contains 72 cities, but individual analyses may contain fewer cities.
- The supplied notebooks do not establish a complete implementation of every result in the v5 manuscript. In particular, the tree-share residual regression and land-cover carbon–water accounting panels are not fully reproduced by this extraction. Publication figure layouts are also outside its scope.

Validation so far includes syntax and import checks, command-line help, and synthetic-data checks of the tree-contribution and flux-robustness workflows. A full run against the original research data has not been performed.

## Citation

When referring to the accompanying manuscript, use its current title and author list:

> Zhu, S., Dong, W., Xu, J., Gharun, M., Zheng, Z., Wang, L., Chen, M. & Dash, J. *Disproportionate tree carbon uptake and sustainable urban greening beyond green space area*. Manuscript in preparation.

Update this citation with the final publication details and add a versioned code archive DOI when those records become available. The Zenodo DOI above refers to the data record identified in the manuscript and should not be presented as an archived code DOI.
