#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Urban UFLUX: analysis-only extraction for code availability.

Source notebooks
----------------
Urban_FLUX_v3.ipynb (primary) and Urban_FLUX_analysis.ipynb (comparison).
Shared LUCC adjustment and Moran functions were compared across both sources;
older UFLUXv2 workflows are superseded by v3's UFLUXv1 analysis. This is a
curated extraction, not a new implementation of the paper's methods.
Original source cell numbers below are zero-based.

Usage
-----
python urban_uflux_analysis.py --help
python urban_uflux_analysis.py --data-root /path/to/urban_sust --stage tree
python urban_uflux_analysis.py --data-root /path/to/urban_sust --stage robustness
python urban_uflux_analysis.py --data-root /path/to/urban_sust --stage summary
python urban_uflux_analysis.py --data-root /path/to/urban_sust --stage tables
python urban_uflux_analysis.py --data-root /path/to/urban_sust --stage drivers
python urban_uflux_analysis.py --data-root /path/to/urban_sust --stage ablation
python urban_uflux_analysis.py --data-root /path/to/urban_sust --stage spatial
python urban_uflux_analysis.py --data-root /path/to/urban_sust --stage socioeconomic

Dependencies (install separately; this script never installs packages)
-------------------------------------------------------------------
Python >=3.10; numpy, pandas, scipy, xarray, netCDF4, geopandas, rasterio,
rioxarray, pyproj, scikit-learn, xgboost, statsmodels, joblib, tqdm.
Summary/spatial also require libpysal and esda. ERA5 and socioeconomic raster
loading uses the source notebooks' geoxai dependency:
    python -m pip install git+https://github.com/songyanzhu/geoxai.git
The original package versions were not recorded in the supplied notebooks;
no claim of a fully pinned reproduction environment is made.

Required existing inputs (not distributed in this Python file)
-------------------------------------------------------------
<data-root>/2_geojsons/<city>.geojson
<data-root>/4_analysis/spatial_EVI_DynamicWorld_UFLUXv1/<city>_uflux.nc
  Annual time, latitude, longitude coordinates; 2016--2025.
  NEP/GPP/RECO/NEP_dir: g C m-2 d-1; ET: mm yr-1; LC: Dynamic World 0--8.
  Tree analysis needs NEP and LC. Robustness additionally uses GPP, RECO,
  EVI, and optional NEP_dir; see analyse_city() for scenario requirements.
<data-root>/1_griddata/<city>/<city>_ERA5_01deg/*.tif
  Or the original fallback 1_griddata_202605/<city>/... . Required for
  summary/spatial/socioeconomic. Band ordering is in load_era5().
Tables reads existing summary CSVs or outputs of --stage summary.
Drivers/ablation read existing summary joblib data or outputs of --stage summary.
Socioeconomic also needs 1_Worldpop/<ISO3>_<product>_<year>.tif and city-level
Aqua_temperature_1km, Terra_temperature_1km, VIIRS_Nightlight_500m subfolders.
Outputs are written beneath data-root, retaining the original folder names.
Only load trusted joblib files (joblib uses Python pickle).

Extraction boundaries and interpretation
----------------------------------------
Includes period means, cover transitions, unchanged-initial-class adjustment,
city summaries, exploratory linear/Ridge/XGBoost attribution, Moran/LISA,
area-normalised centre/periphery LISA, apparent-vs-exogenous ratio ablation,
tree positive-NEP contributions, flux robustness, population-normalised NEP
and nighttime-light co-trends. Excludes GEE downloads, satellite retrieval,
flux-model training/upscaling, GenAI calls, obsolete/pilot blocks, duplicate
plots, and notebook display-only cells. No model training for flux production
is included; driver regressions are part of the retained statistical analysis.

Important source limitations retained for scientific traceability:
* get_ridge_reg() reports IN-SAMPLE R2/RMSE; RidgeCV selects alpha but those
  metrics are not held-out performance. run_linear() scales before splitting.
* Original ablation uses within-sample importance, zero filling and independent
  subsamples; its LUE/IWUE formula and units require author verification before
  treating it as a validated mechanistic implementation. No silent scientific
  corrections have been made to these formulas or their reported estimands.
* Legacy moran_list() uses the full-period mean and angular radial distances;
  --stage spatial separately uses 2016--2020 baseline and metric coordinates.
  Original spatial robustness defaults (200 m, 99 permutations) are retained;
  these differ from any manuscript settings specifying 999 permutations.
* Original city exclusions (e.g. Dakar/Accra) remain in the relevant workflows.
* The supplied notebooks do not establish a complete implementation of every
  v5 manuscript result, notably the tree-share residual regression and the
  land-cover carbon/water accounting panels. This file does not invent missing
  analyses or claim exact reproduction of all manuscript figures/tables.
* Source parameter choices, masks and units are retained. Portability edits:
  configurable paths; explicit imports; function/CLI boundaries; CSV exports;
  deterministic NumPy seed; zero-cluster division guards; failures propagate.
  Existing output caching remains as in the sources: use a clean data copy
  when a complete recalculation is needed. Actual source data were not supplied,
  so full numerical reproduction must be checked against the authors' data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings


def _load_dependencies(stage):
    # Lazy loading keeps --help usable without the scientific environment.
    global np, pd, xr, gpd, joblib, tqdm, Geod, Transformer
    global geometry_mask, from_bounds, geoface, saturation_vapor_pressure
    global StandardScaler, train_test_split, LinearRegression, Ridge, RidgeCV
    global Pipeline, r2_score, mean_squared_error, XGBRegressor
    global permutation_importance, variance_inflation_factor, libpysal, Moran, Moran_Local
    import numpy as np
    import pandas as pd
    import xarray as xr
    import geopandas as gpd
    import rioxarray  # registers the .rio accessor
    import joblib
    from tqdm import tqdm
    from pyproj import Geod, Transformer
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds
    if stage in {'summary', 'drivers', 'ablation'}:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import r2_score, mean_squared_error
        from sklearn.inspection import permutation_importance
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from xgboost import XGBRegressor
    if stage in {'summary', 'spatial', 'socioeconomic'}:
        from geoxai.geo import geoface
        from geoxai.geo.meteo import saturation_vapor_pressure
    if stage in {'summary', 'spatial'}:
        import libpysal
        from esda.moran import Moran, Moran_Local



def _init_metadata():
    global meta_city, tier_info, city_continents, d_dw_lc
    # Source: v3 cell 2 (metadata only; online geocoding removed).
    meta_city = pd.DataFrame([
        # Alpha++ (global command centers)
        ("London", "Alpha++", "Europe", "United Kingdom", "Global financial, political, and cultural hub"),
        ("New York", "Alpha++", "Americas", "USA", "Global financial, political, and cultural hub"),

        # Alpha+ (top-tier global hubs)
        ("Tokyo", "Alpha+", "Asia", "Japan", "Major economic, cultural, and technological hub"),
        ("Paris", "Alpha+", "Europe", "France", "Major cultural, political, and economic center"),
        ("Singapore", "Alpha+", "Asia", "Singapore", "Global finance, transport, and trade hub"),
        ("Hong Kong", "Alpha+", "Asia", "Hong Kong", "Global finance and trade hub"),
        ("Shanghai", "Alpha+", "Asia", "China", "Major economic and financial center"),
        ("Beijing", "Alpha+", "Asia", "China", "Political and cultural capital"),
        ("Dubai", "Alpha+", "Middle East", "UAE", "Business and tourism hub"),

        # Alpha - Americas
        ("Los Angeles", "Alpha", "Americas", "USA", "Entertainment and economic center"),
        ("Chicago", "Alpha", "Americas", "USA", "Financial and industrial hub"),
        ("Toronto", "Alpha", "Americas", "Canada", "Economic and cultural hub"),
        ("Mexico City", "Alpha", "Americas", "Mexico", "Political, economic, and cultural center"),
        ("Sao Paulo", "Alpha", "Americas", "Brazil", "Major economic center"),

        # Alpha - Europe
        ("Madrid", "Alpha", "Europe", "Spain", "Political and cultural center"),
        ("Milan", "Alpha", "Europe", "Italy", "Fashion and financial hub"),
        ("Frankfurt", "Alpha", "Europe", "Germany", "Financial and transport hub"),
        ("Amsterdam", "Alpha", "Europe", "Netherlands", "Financial, transport, and cultural hub"),

        # Alpha - Asia-Pacific
        ("Seoul", "Alpha", "Asia", "South Korea", "Economic, technological, and cultural hub"),
        ("Sydney", "Alpha", "Oceania", "Australia", "Economic and cultural center"),
        ("Mumbai", "Alpha", "Asia", "India", "Financial and commercial center"),
        ("Istanbul", "Alpha", "Europe", "Turkey", "Economic, cultural, and historical hub"),

        # Beta+ (strong regional/global connectors)
        ("Bangkok", "Beta+", "Asia", "Thailand", "Regional economic and transport hub"),
        ("Jakarta", "Beta+", "Asia", "Indonesia", "Economic and political center"),
        ("Kuala Lumpur", "Beta+", "Asia", "Malaysia", "Economic and cultural center"),
        ("Manila", "Beta+", "Asia", "Philippines", "Political and economic hub"),
        ("Buenos Aires", "Beta+", "Americas", "Argentina", "Cultural and economic hub"),
        ("Santiago", "Beta+", "Americas", "Chile", "Economic and political center"),
        ("Lima", "Beta+", "Americas", "Peru", "Economic and political hub"),
        ("Bogota", "Beta+", "Americas", "Colombia", "Political and economic center"),
        ("Johannesburg", "Beta+", "Africa", "South Africa", "Economic and industrial hub"),
        ("Cairo", "Beta+", "Africa", "Egypt", "Political and cultural center"),
        ("Nairobi", "Beta+", "Africa", "Kenya", "Economic and transport hub"),
        ("Tel Aviv", "Beta+", "Middle East", "Israel", "Economic and tech hub"),
        ("Vienna", "Beta+", "Europe", "Austria", "Cultural and political center"),
        ("Brussels", "Beta+", "Europe", "Belgium", "Political and administrative hub"),
        ("Zurich", "Beta+", "Europe", "Switzerland", "Financial hub"),
        ("Stockholm", "Beta+", "Europe", "Sweden", "Political and economic center"),
        ("Copenhagen", "Beta+", "Europe", "Denmark", "Economic and cultural hub"),
        ("Dublin", "Beta+", "Europe", "Ireland", "Economic and tech hub"),

        # Beta (important regional hubs)
        ("Hanoi", "Beta", "Asia", "Vietnam", "Political and economic center"),
        ("Ho Chi Minh City", "Beta", "Asia", "Vietnam", "Economic hub"),
        ("Karachi", "Beta", "Asia", "Pakistan", "Economic and port hub"),
        ("Dhaka", "Beta", "Asia", "Bangladesh", "Political and economic center"),
        ("Colombo", "Beta", "Asia", "Sri Lanka", "Economic and administrative center"),
        ("Riyadh", "Beta", "Middle East", "Saudi Arabia", "Political and economic center"),
        ("Doha", "Beta", "Middle East", "Qatar", "Economic and administrative hub"),
        ("Kuwait City", "Beta", "Middle East", "Kuwait", "Political and financial hub"),
        ("Casablanca", "Beta", "Africa", "Morocco", "Economic and financial center"),
        ("Accra", "Beta", "Africa", "Ghana", "Political and economic hub"),
        ("Addis Ababa", "Beta", "Africa", "Ethiopia", "Political and diplomatic center"),
        ("Kinshasa", "Beta", "Africa", "DR Congo", "Political and economic hub"),
        ("Lagos", "Beta", "Africa", "Nigeria", "Economic and commercial hub"),
        ("Warsaw", "Beta", "Europe", "Poland", "Political and economic center"),
        ("Prague", "Beta", "Europe", "Czech Republic", "Cultural and political hub"),
        ("Budapest", "Beta", "Europe", "Hungary", "Cultural and political hub"),
        ("Athens", "Beta", "Europe", "Greece", "Cultural and political hub"),
        ("Lisbon", "Beta", "Europe", "Portugal", "Cultural and economic center"),

        # Gamma (emerging global cities)
        ("Dar es Salaam", "Gamma", "Africa", "Tanzania", "Emerging economic hub"),
        ("Kampala", "Gamma", "Africa", "Uganda", "Emerging economic and political hub"),
        ("Dakar", "Gamma", "Africa", "Senegal", "Emerging economic and cultural hub"),
        ("Phnom Penh", "Gamma", "Asia", "Cambodia", "Emerging economic and political hub"),
        ("Yangon", "Gamma", "Asia", "Myanmar", "Emerging economic hub"),
        ("Kathmandu", "Gamma", "Asia", "Nepal", "Emerging political and cultural hub"),
        ("Quito", "Gamma", "Americas", "Ecuador", "Emerging economic and cultural hub"),
        ("La Paz", "Gamma", "Americas", "Bolivia", "Emerging political hub"),
        ("Montevideo", "Gamma", "Americas", "Uruguay", "Emerging political and economic hub"),

        # Optional / niche
        ("Reykjavik", "Optional", "Europe", "Iceland", "Subpolar city"),
        ("Anchorage", "Optional", "Americas", "USA", "High latitude city"),
        ("Perth", "Optional", "Oceania", "Australia", "Mediterranean/arid edge city"),
        ("Vancouver", "Optional", "Americas", "Canada", "Temperate rainforest city"),
        ("Phoenix", "Optional", "Americas", "USA", "Hot desert city"),
    ], columns=["city", "tier", "region", "country", "notes"]).set_index("city")

    tier_info = {
        "Alpha++": "Global command centers: cities with worldwide influence in finance, politics, and culture.",
        "Alpha+": "Top-tier global hubs: cities with strong global economic, cultural, or political influence.",
        "Alpha": "Major global cities: significant economic and cultural centers with global connectivity.",
        "Beta+": "Strong regional/global connectors: cities that connect regions and influence beyond their borders.",
        "Beta": "Important regional hubs: key economic, political, or cultural centers within their region.",
        "Gamma": "Emerging global cities: growing urban centers important for regional representation.",
        "Optional": "Special/niche cities: included for specific geographic or climate diversity."
    }

    meta_city.loc['Istanbul', 'region'] = 'Europe'

    city_continents = dict(zip(meta_city.index, meta_city["region"]))

    d_dw_lc = {
        0: 'water',
        1: 'trees',
        2: 'grass',
        3: 'flooded_vegetation',
        4: 'crops',
        5: 'shrub_and_scrub',
        6: 'built',
        7: 'bare',
        8: 'snow_and_ice'
    }


# Source: v3 cell 5 (analysis input helpers only).
def load_era5(p_ERA5):
    era5 = geoface.load_tif(p_ERA5, band_names = [
        'dewpoint_temperature_2m', 'temperature_2m', 'soil_temperature_level_1', 'snow_cover',
        'volumetric_soil_water_layer_1', 'forecast_albedo', 'surface_latent_heat_flux_sum', 'surface_sensible_heat_flux_sum',
        'surface_solar_radiation_downwards_sum', 'surface_thermal_radiation_downwards_sum',
        'surface_net_solar_radiation_sum', 'surface_net_thermal_radiation_sum',
        'evaporation_from_the_top_of_canopy_sum', 'surface_pressure', 'total_precipitation_sum',
        'temperature_2m_min', 'temperature_2m_max', 'u_component_of_wind_10m', 'v_component_of_wind_10m',
    ])
    era5 = era5.where(era5 != 0)
    era5 = fill_era5_na_nearest(era5)
    data_vars = era5.data_vars
    for v in ['dewpoint_temperature_2m', 'temperature_2m', 'soil_temperature_level_1', 'temperature_2m_min', 'temperature_2m_max']:
        if v in data_vars:
            era5[v] -= 273.15
    for v in [
        'surface_latent_heat_flux_sum', 'surface_sensible_heat_flux_sum', 'surface_solar_radiation_downwards_sum',
        'surface_thermal_radiation_downwards_sum', 'surface_net_solar_radiation_sum', 'surface_net_thermal_radiation_sum'
        ]:
        if v in data_vars:
            era5[v] /= 86400

    if 'surface_pressure' in data_vars:
        era5['surface_pressure'] /= 100

    era5['VPD'] = saturation_vapor_pressure(era5['temperature_2m']) - saturation_vapor_pressure(era5['dewpoint_temperature_2m'])
    if ('surface_net_solar_radiation_sum' in data_vars) and ('surface_net_thermal_radiation_sum' in data_vars):
        era5['surface_net_radiation_sum'] = era5['surface_net_solar_radiation_sum'] + era5['surface_net_thermal_radiation_sum']
        era5 = era5.drop_vars(['surface_net_solar_radiation_sum', 'surface_net_thermal_radiation_sum'])
    era5 = era5.drop_vars('dewpoint_temperature_2m')
    # era5 = era5.drop_vars('spatial_ref')
    era5 = era5.drop_vars([
        'forecast_albedo', 'surface_latent_heat_flux_sum', 'surface_sensible_heat_flux_sum',
        'evaporation_from_the_top_of_canopy_sum', 'temperature_2m_min', 'temperature_2m_max',
        'u_component_of_wind_10m', 'v_component_of_wind_10m', 'snow_cover'
    ])
    return era5

def mode_along_time(a):
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    return np.bincount(a.astype(int)).argmax()

def get_pixel_area_km2(nc):
    pixel_area = np.abs(float(
        (nc.longitude[0] - nc.longitude[1]) * \
        1e5 * (nc.latitude[0] - nc.latitude[1]) * 1e5 / 1e6
    ))  # km2

    return pixel_area

def fill_era5_na_nearest(ds, x = 'longitude', y = 'latitude'):
    from scipy.interpolate import griddata
    filled_ds = ds.copy()

    for var in ds.data_vars:  # Loop through each variable in the Dataset
        da = ds[var]  # Extract DataArray
        X, Y = np.meshgrid(da[x], da[y])

        # Get valid (non-NaN) points
        valid_mask = ~np.isnan(da.values)
        valid_points = np.column_stack([X[valid_mask], Y[valid_mask]])
        valid_values = da.values[valid_mask]

        # Interpolate NaNs using nearest neighbor
        filled_values = griddata(valid_points, valid_values, (X, Y), method="nearest")

        # Assign filled values back to the dataset
        filled_ds[var] = xr.DataArray(filled_values, coords=da.coords, dims=da.dims)

    return filled_ds

# Source: v3 cell 6.
def get_LUCC_caused_changes(ncu0_avg, ncu1_avg):
    '''
    dLUCC = dObs - dClimate
    For example:
    The target pixel is grassland in the first period.
    If there were no land-use change, it should have changed like a grassland.
    '''

    # -----------------------------
    # 0. Numeric variables dynamically
    # -----------------------------
    vars_num = [
        v for v in ncu0_avg.data_vars
        if v in ncu1_avg.data_vars
        and v != 'LC'
        and np.issubdtype(ncu0_avg[v].dtype, np.number)
    ]


    # -----------------------------
    # 1. LUCC mask
    # -----------------------------
    luc_mask = ncu0_avg['LC'] != ncu1_avg['LC']
    no_luc_mask = ~luc_mask

    # -----------------------------
    # 2. Observed changes
    # -----------------------------
    delta_obs = ncu1_avg[vars_num] - ncu0_avg[vars_num]

    # -----------------------------
    # 3. Climate signal per LC
    # -----------------------------
    climate_signal = {}

    for var in vars_num:
        clim = delta_obs[var].where(no_luc_mask).groupby(ncu0_avg['LC']).mean()
        climate_signal[var] = clim

    # -----------------------------
    # 4. LUCC contribution (robust mapping)
    # -----------------------------
    delta_lucc = xr.Dataset()

    for var in vars_num:
        clim_da = climate_signal[var]

        # If clim_da is scalar → broadcast to all pixels
        if np.isscalar(clim_da):
            delta_clim_pixel = xr.ones_like(ncu0_avg['LC']) * clim_da
        else:
            # Pixel-wise mapping using indexing: DataArray.values → dict
            clim_dict = dict(zip(clim_da['LC'].values, clim_da.values))
            # Map each pixel based on original LC
            delta_clim_pixel = xr.DataArray(
                np.vectorize(lambda x: clim_dict.get(x, np.nan))(ncu0_avg['LC'].values),
                dims=ncu0_avg['LC'].dims,
                coords=ncu0_avg['LC'].coords
            )

        # LUCC-only Δ
        delta_lucc[var] = delta_obs[var].where(luc_mask) - delta_clim_pixel

    # -----------------------------
    # 5. Transition label
    # -----------------------------
    transition = xr.DataArray(
        ncu0_avg['LC'].astype(str).values + "→" + ncu1_avg['LC'].astype(str).values,
        dims=ncu0_avg['LC'].dims,
        coords=ncu0_avg['LC'].coords
    )
    delta_lucc = delta_lucc.assign_coords(transition=transition)

    return delta_lucc

# Source: v3 cell 7.
def load_city_uflux(city_name, suffix, folder_suffix = '', period1 = ['2016', '2020'], period2 = ['2021', '2025']):
    shp = gpd.read_file(root_proj.joinpath(f"2_geojsons/{city_name}.geojson"), driver='GeoJSON')

    ncu = xr.open_dataset(root_proj.joinpath(f'4_analysis{folder_suffix}/spatial{suffix}').joinpath(f'{city_name}_uflux.nc'))
    ncu = ncu.assign_coords(time = [pd.Timestamp(t).replace(month=1) for t in ncu.time.values])
    ncu = ncu.rio.write_crs("epsg:4326", inplace = False).rio.clip(shp.geometry.values, shp.crs)
    ncu = ncu.sel(time = slice('2016', '2025'))
    # ncu['LC'] = xr.apply_ufunc(lambda x: np.floor(x + 0.5), ncu['LC'])
    # ncu['LC'] = ncu['LC'].where(ncu['LC'] != 3, 2)
    # lc = xr.open_dataset(root_proj.joinpath(f'4_analysis/spatial{suffix}').joinpath(f'{city_name}_ESA_LC.nc'))['LC']
    # ncu = ncu.where(~ncu['GPP'].isnull() & ~ncu['RECO'].isnull() & ~ncu['ET'].isnull() & ~ncu['EVI'].isnull(), np.nan)
    for k in ncu.keys(): ncu[k].attrs.pop('long_name', None)

    # ------------------------------------------------------------------------------

    if root_proj.joinpath(f'1_griddata/{city_name}').exists():
        city_folder = root_proj.joinpath(f'1_griddata/{city_name}')
    else:
        city_folder = root_proj.joinpath(f'1_griddata_202605/{city_name}')

    df_path_ERA5 = []
    for p in city_folder.joinpath(f'{city_name}_ERA5_01deg').glob('*.tif'):
        dt = pd.to_datetime(p.stem)
        df_path_ERA5.append([dt, p])
    df_path_ERA5 = pd.DataFrame(df_path_ERA5, columns = ['YEAR', 'PATH']).set_index('YEAR').sort_index()

    df_path_ERA5 = df_path_ERA5.loc[ncu.time, :]

    nc_era5 = []
    for dt in tqdm(df_path_ERA5.index):
        p = df_path_ERA5.loc[dt, 'PATH']
        era5t = load_era5(p)
        era5t = era5t.rename({
            'temperature_2m': 'T2m',
            'soil_temperature_level_1': 'Tsoil1',
            'volumetric_soil_water_layer_1': 'Swvl1',
            'surface_solar_radiation_downwards_sum': 'SWIN',
            'surface_thermal_radiation_downwards_sum': 'LWIN',
            'surface_net_radiation_sum': 'Rnet',
            'surface_pressure': 'PA',
            'total_precipitation_sum': 'P',
            'VPD': 'VPD'
        }).expand_dims(time = [dt])
        nc_era5.append(era5t); del(era5t)
    nc_era5 = xr.merge(nc_era5)

    ncu = xr.merge([
        ncu,
        nc_era5.interp(latitude = ncu.latitude, longitude = ncu.longitude).rio.write_crs("epsg:4326", inplace = False).rio.clip(shp.geometry.values, shp.crs)
    ]).drop_vars('spatial_ref')

    ncu['ET'] = ncu['ET'].clip(min = 1e-9, max = 3000)

    # calc ecosystem water retention: ET is mm/year and precip P is annual averaged m
    # Rw is unitless
    ncu['Rw'] = ncu['ET'] / (ncu['P'] * 365 * 1000)
    # calc LUE: SWIN is in Wm-2, x 0.0864 => MJm−2day−1
    # LUE unit is gC/MJ
    ncu['fAPAR'] = (ncu['EVI'] * 1.25).clip(0, 1)
    PAR = 0.45 * ncu['SWIN'] * 0.0864
    APAR = PAR * ncu['fAPAR']
    ncu['LUE'] = ncu['GPP'] / APAR
    ncu['LUE'] = ncu['LUE'].clip(min = 0, max = 5)
    # calc WUE: ET is mm/year = 1 kg H₂O m−2; GPP is annual averaged gC m-2 d-1
    # WUE unit: gC/mm (=gC/kgH2​O)
    ncu['WUE'] = (ncu['GPP'] * 365) / ncu['ET']
    ncu['WUE'] = ncu['WUE'].clip(min = 0, max = 10)

    # get multiyear average
    ncu_avg = ncu.mean(dim = 'time')
    ncu_avg['LC'] = xr.apply_ufunc(
        mode_along_time,
        ncu['LC'],
        input_core_dims=[['time']],   # reduce over time
        vectorize=True,               # loop over all other dims
        dask='parallelized',          # works with dask-backed arrays
        output_dtypes=[float],
    )

    del(nc_era5)

    # ==============================================================================
    # Before and after

    # First five years
    # ncu0 = ncu.isel(time = slice(0, 5))
    ncu0 = ncu.sel(time = slice(period1[0], period1[1]))
    # Last fice years
    # ncu1 = ncu.isel(time = slice(-6, -1))
    ncu1 = ncu.sel(time = slice(period2[0], period2[1]))

    ncu0_avg = ncu0.mean(dim = 'time')
    ncu0_avg['LC'] = xr.apply_ufunc(
        mode_along_time,
        ncu0['LC'],
        input_core_dims=[['time']],   # reduce over time
        vectorize=True,               # loop over all other dims
        dask='parallelized',          # works with dask-backed arrays
        output_dtypes=[float],
    )


    ncu1_avg = ncu1.mean(dim = 'time')
    ncu1_avg['LC'] = xr.apply_ufunc(
        mode_along_time,
        ncu1['LC'],
        input_core_dims=[['time']],   # reduce over time
        vectorize=True,               # loop over all other dims
        dask='parallelized',          # works with dask-backed arrays
        output_dtypes=[float],
    )



    # ==========================================================================

    # @title Get difference between two periods

    ncd = (ncu1_avg - ncu0_avg).drop_vars('LC')

    lucc_code = ncu0_avg['LC'] * 10 + ncu1_avg['LC']
    lucc_code = np.char.zfill(lucc_code.fillna(99).astype(int).astype(str), 2)
    # print(np.unique(lucc_code))

    # 0: water
    # 1: trees
    # 2: grass
    # 3: flooded_vegetation
    # 4: crops
    # 5: shrub_and_scrub
    # 6: built
    # 7: bare
    # 8: snow_and_ice

    lucc_categories = {
        'Unchanged': ['00', '03', '11', '22', '30', '33', '44', '55', '66', '77', '88'],
        'Urbanisation': ['06', '16', '26', '36', '46', '56', '76', '86'],
        'Agricultural expansion': ['04', '14', '24', '34', '54', '64', '74', '84'],
        'Agricultural abandonment': ['42', '45', '47'],
        'Flooding / inundation': ['10', '13', '20', '23', '40', '50', '53', '60', '63', '70', '73'],
        'Greening': ['01', '02', '05', '21', '31', '32', '35', '41', '51', '61', '62', '65', '71', '72', '75', '81', '82', '85'],
        'Degradation': ['15', '17', '18', '27', '28', '57'],
        'Other change': [
            '43', # crops -> paddy fields
            '80', # snow/ice -> water
            '83', # snow/ice -> flooded veg
            '07', # water -> bare
            '08', # water -> snow/ice
            '12', # tree -> grass
            '25', # grass -> shrub
            '37', # flooded veg -> bare
            '38', # flooded veg -> snow/ice
            '48', # crop -> snow/ice
            '52', # shurb -> grass
            '58', # shurb -> snow/ice
            '67', # built -> bare
            '68', # built -> snow/ice
            '78', # bare -> snow/ice
            '87' # snow/ice -> bare
        ]
    }

    lucc = np.full(ncu_avg['LC'].shape, np.nan)
    lucc[np.isin(lucc_code, lucc_categories['Unchanged'])] = 0 # Unchanged
    lucc[np.isin(lucc_code, lucc_categories['Urbanisation'])] = 1 # Urbanisation
    lucc[np.isin(lucc_code, lucc_categories['Agricultural expansion'])] = 2 # Agricultural expansion
    lucc[np.isin(lucc_code, lucc_categories['Agricultural abandonment'])] = 3 # Agricultural abandonment
    lucc[np.isin(lucc_code, lucc_categories['Flooding / inundation'])] = 4 # Flooding / inundation
    lucc[np.isin(lucc_code, lucc_categories['Greening'])] = 5 # Greening (incl. afforestation)
    lucc[np.isin(lucc_code, lucc_categories['Degradation'])] = 6 # Degradation / clearing
    lucc[np.isin(lucc_code, lucc_categories['Other change'])] = 7 # Other change

    # appointed_list = []
    # for v in lucc_categories.values():
    #     appointed_list += v
    # # print(appointed_list)

    # full_list = []
    # for i in range(9):
    #     for j in range(9):
    #         full_list.append(f'{i}{j}')
    # # full_list.append('99')

    # # print(full_list)
    # print('Unassigned changes:', [x for x in full_list if x not in appointed_list])
    # print('Duplicated:', [item for item in appointed_list if appointed_list.count(item) > 1])

    ncd['LUCC'] = (
        ('latitude', 'longitude'),
        lucc
    )

    ncd_lucc = get_LUCC_caused_changes(ncu0_avg, ncu1_avg)
    ncd_lucc['LUCC'] = ncd['LUCC']

    return {
        'uflux': ncu,
        'uflux_avg': ncu_avg,
        'uflux_period0': ncu0,
        'uflux_period0_avg': ncu0_avg,
        'uflux_period1': ncu1,
        'uflux_period1_avg': ncu1_avg,
        'uflux_periods_diff': ncd,
        'uflux_periods_diff_lucc': ncd_lucc,
    }

# Source: v3 cell 8.
def moran_list(da):
    '''
    Moran's I > 0 -> clustering
    Moran's I appox. 0 -> random
    Moran's I <0 -> dispersion

    HH: increase
    LL: decrease
    '''

    def _count(region, label):
        return ((cluster_map == label) & region).sum().item()

    data = da.values

    # --- Flatten & mask ---
    data_flat = data.flatten()
    mask_flat = np.isfinite(data_flat)
    y = data_flat[mask_flat]

    # --- Build weights for full grid ---
    w_full = libpysal.weights.lat2W(*data.shape)

    # --- Manually subset weights (version-safe) ---
    valid_ids = np.where(mask_flat)[0]
    valid_id_set = set(valid_ids)

    neighbors = {i: [n for n in w_full.neighbors[i] if n in valid_id_set]
                for i in valid_ids}

    weights = {i: [1] * len(neighbors[i]) for i in valid_ids}

    w = libpysal.weights.W(neighbors, weights, silence_warnings=True)

    # ------------------------------------------------------------------------------

    # --- Moran's I ---
    moran = Moran(y, w)

    result_summary_moran = {
        "Moran_I": float(round(moran.I, 4)),
        "p_value": float(round(moran.p_sim, 4)),
        "N_valid_pixels": int(len(y))
    }

    # ------------------------------------------------------------------------------

    # --- Local Moran ---
    lisa = Moran_Local(y, w)

    # Significant clusters only
    sig = lisa.p_sim < 0.05
    quadrant = lisa.q  # 1 HH, 2 LH, 3 LL, 4 HL

    cluster = np.full(data_flat.shape, np.nan)
    cluster[mask_flat] = np.where(sig, quadrant, np.nan)

    cluster_map = xr.DataArray(
        cluster.reshape(data.shape),
        coords=da.coords,
        dims=da.dims
    )

    # --- Midpoints (spatial reference) ---
    lat_mid = cluster_map.latitude.mean()
    lon_mid = cluster_map.longitude.mean()

    # --- Quadrant masks ---
    NE = (cluster_map.latitude > lat_mid) & (cluster_map.longitude > lon_mid)
    NW = (cluster_map.latitude > lat_mid) & (cluster_map.longitude <= lon_mid)
    SE = (cluster_map.latitude <= lat_mid) & (cluster_map.longitude > lon_mid)
    SW = (cluster_map.latitude <= lat_mid) & (cluster_map.longitude <= lon_mid)

    # --- Counts ---
    stats = {
        "NE_HH": _count(NE, 1),
        "NW_HH": _count(NW, 1),
        "SE_HH": _count(SE, 1),
        "SW_HH": _count(SW, 1),

        "NE_LL": _count(NE, 3),
        "NW_LL": _count(NW, 3),
        "SE_LL": _count(SE, 3),
        "SW_LL": _count(SW, 3),
    }

    # --- Totals ---
    total_HH = (cluster_map == 1).sum().item()
    total_LL = (cluster_map == 3).sum().item()

    # --- Percentages ---
    percent = {
        k: (v / total_HH * 100 if "HH" in k and total_HH else
            v / total_LL * 100 if "LL" in k and total_LL else np.nan)
        for k, v in stats.items()
    }

    # --- Centre vs periphery ---
    lat0 = cluster_map.latitude.mean()
    lon0 = cluster_map.longitude.mean()

    dist = np.sqrt((cluster_map.latitude - lat0)**2 +
                (cluster_map.longitude - lon0)**2)

    threshold = np.nanpercentile(dist, 30)  # inner 30% = centre

    centre = dist <= threshold
    periphery = dist > threshold

    centre_stats = {
        "centre_HH_%": ((cluster_map == 1) & centre).sum().item() / total_HH * 100 if total_HH else np.nan,
        "periphery_HH_%": ((cluster_map == 1) & periphery).sum().item() / total_HH * 100 if total_HH else np.nan,
        "centre_LL_%": ((cluster_map == 3) & centre).sum().item() / total_LL * 100 if total_LL else np.nan,
        "periphery_LL_%": ((cluster_map == 3) & periphery).sum().item() / total_LL * 100 if total_LL else np.nan,
    }

    result_summary_lisa = {
        "quadrant_counts": {k: int(v) for k, v in stats.items()},
        "quadrant_percentages": {k: float(v) for k, v in percent.items()},
        "centre_vs_periphery_percent": {k: float(v) for k, v in centre_stats.items()}
    }

    return result_summary_moran, result_summary_lisa

# Source: v3 cell 9.
def run_linear(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)

    y_pred = model_linear.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print("R²:", r2)
    # print("RMSE:", rmse)

    importance = pd.Series(model_linear.coef_, index=X.columns)
    importance = importance.sort_values(key=abs, ascending=False)

    return {
        'model': model_linear,
        'r2': r2,
        'rmse': rmse,
        'importance': importance,
    }

def run_xgboost(X, y):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # XGBoost model
    model_xgb = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print("R²:", r2)
    # print("RMSE:", rmse)

    # Permutation importance
    result = permutation_importance(
        model_xgb, X_test, y_test,
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )

    perm_importance = pd.Series(result.importances_mean, index=X.columns)
    perm_importance = perm_importance.sort_values(ascending=False)

    return {
        'model': model_xgb,
        'r2': r2,
        'rmse': rmse,
        'importance': perm_importance,
    }

# Source: v3 cell 77.
def get_ridge_reg(dd_ml, y_name, X_names = ['EVI', 'T2m', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE', 'WUE']):
    # ----------------------------
    # 1️⃣ Select features and target
    # ----------------------------
    # X_names = ['EVI', 'T2m', 'Tsoil1', 'Swvl1', 'SWIN', 'LWIN', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE', 'WUE']
    # X_names = ['EVI', 'T2m', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE', 'WUE']
    X = dd_ml['data'][X_names]
    y = dd_ml['data'][y_name]

    # ----------------------------
    # 2️⃣ Check multicollinearity with VIF
    # ----------------------------
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])]

    # print("VIF (Variance Inflation Factor) sorted:")
    # print(vif.sort_values("VIF", ascending=False))

    # ----------------------------
    # 3️⃣ Ridge regression with cross-validation to select best alpha
    # ----------------------------
    alphas = np.logspace(-3, 3, 10)

    ridge_cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, cv=5))
    ])

    ridge_cv_pipeline.fit(X, y)
    best_alpha = ridge_cv_pipeline.named_steps["ridge"].alpha_
    # print("\nBest alpha selected by CV:", best_alpha)

    # Fit final Ridge model with best alpha
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha))
    ])
    ridge_pipeline.fit(X, y)

    # ----------------------------
    # 4️⃣ Predict and calculate performance metrics
    # ----------------------------
    y_pred = ridge_pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    metrics = {"R2": r2, "RMSE": rmse}
    # print("\nModel performance metrics:", metrics)

    # ----------------------------
    # 5️⃣ Standardized coefficients (unitless)
    # ----------------------------
    beta_std = pd.Series(ridge_pipeline.named_steps["ridge"].coef_, index=X_names)

    # ----------------------------
    # 6️⃣ Convert back to actual effect (original units)
    # ----------------------------
    sigma_X = X.std()
    sigma_y = y.std()
    beta_actual = beta_std * (sigma_y / sigma_X)

    # ----------------------------
    # 7️⃣ Combine standardized, actual effect, and VIF into a single DataFrame
    # ----------------------------
    coef_vif = pd.DataFrame({
        "Variable": X_names,
        "beta_std": beta_std,
        "beta_actual": beta_actual
    }).merge(vif, on="Variable")

    # Optional: sort by absolute actual effect
    coef_vif = coef_vif.sort_values(by="beta_actual", key=abs, ascending=False).reset_index(drop=True)
    coef_vif = coef_vif.set_index('Variable')
    coef_vif.index.name = ''

    # ----------------------------
    # 8️⃣ Return as dictionary
    # ----------------------------
    results_dict = {
        "r2": r2,
        "rmse": rmse,
        "alpha": best_alpha,
        "coef_vif": coef_vif
    }
    return results_dict

def run_summary():
    'v3 cell 40: period summaries, LUCC adjustment, legacy spatial statistics and driver regressions.'
    suffix = '_EVI_DynamicWorld_UFLUXv1'
    city_names = available_cities()

    for city_name in tqdm(city_names):
        print(city_name)

        savefolder = root_proj.joinpath(f'4_analysis/summary/{city_name}')
        if savefolder.exists(): continue
        savefolder.mkdir(exist_ok = True, parents = True)

        shp = gpd.read_file(root_proj.joinpath(f"2_geojsons/{city_name}.geojson"), driver='GeoJSON')

        period1 = ['2016', '2020']; period2 = ['2021', '2025']

        d_dataset = load_city_uflux(city_name, suffix, period1 = period1, period2 = period2)
        ncu = d_dataset['uflux']
        ncu_avg = d_dataset['uflux_avg']
        ncu0 = d_dataset['uflux_period0']
        ncu0_avg = d_dataset['uflux_period0_avg']
        ncu1 = d_dataset['uflux_period1']
        ncu1_avg = d_dataset['uflux_period1_avg']
        ncd = d_dataset['uflux_periods_diff']
        ncd_lucc = d_dataset['uflux_periods_diff_lucc']


        df_avg = ncu_avg.drop_vars('LC').to_dataframe().dropna()
        area = shp.to_crs(shp.estimate_utm_crs()).geometry.area.values[0] / 1e6
        df_avg = df_avg.mean().rename(f'Avg({period1[0]}-{period2[1]})').to_frame().T
        df_avg['Area_km2'] = area

        df0_avg = ncu0_avg.drop_vars('LC').to_dataframe().dropna()
        area = shp.to_crs(shp.estimate_utm_crs()).geometry.area.values[0] / 1e6
        df0_avg = df0_avg.mean().rename(f'Avg({period1[0]}-{period1[1]})').to_frame().T
        df0_avg['Area_km2'] = area

        df1_avg = ncu1_avg.drop_vars('LC').to_dataframe().dropna()
        area = shp.to_crs(shp.estimate_utm_crs()).geometry.area.values[0] / 1e6
        df1_avg = df1_avg.mean().rename(f'Avg({period2[0]}-{period2[1]})').to_frame().T
        df1_avg['Area_km2'] = area

        df_avg = pd.concat([df_avg, df0_avg, df1_avg], axis = 0)



        df_avg_lc = ncu_avg.to_dataframe().dropna()
        df_avg_lc_count = df_avg_lc.groupby('LC').count()['NEP'].rename('%pixel')

        df_avg_lc = pd.concat([
            df_avg_lc.groupby('LC').mean(),
            df_avg_lc_count / df_avg_lc_count.sum()
        ], axis = 1)



        df_lucc = ncd.to_dataframe().dropna()
        df_lucc_count = df_lucc.groupby('LUCC').count()['NEP'].rename('%pixel')

        df_lucc = pd.concat([
            df_lucc.groupby('LUCC').mean(),
            df_lucc_count / df_lucc_count.sum()
        ], axis = 1)


        df_lucc_sum = ncd.to_dataframe().dropna()
        df_lucc_sum = pd.concat([
            df_lucc_sum.mean().rename('Avg sum').to_frame().T,
            df_lucc_sum[df_lucc_sum['LUCC'] == 0].mean().rename('No change sum').to_frame().T,
            df_lucc_sum[df_lucc_sum['LUCC'] != 0].mean().rename('LUCC sum').to_frame().T,
            ncd_lucc.to_dataframe().dropna().drop(['transition'], axis = 1).mean().rename('LUCC only sum').to_frame().T
        ], axis = 0)


        df_lucc_only = ncd_lucc.to_dataframe().dropna().drop(['transition'], axis = 1).groupby('LUCC').mean()


        df_avg.to_csv(savefolder.joinpath('UFLUX_multiyear_average.csv'))
        df_avg_lc.to_csv(savefolder.joinpath('UFLUX_multiyear_average_LC.csv'))
        df_lucc.to_csv(savefolder.joinpath('UFLUX_multiyear_diff_LUCC.csv'))
        df_lucc_sum.to_csv(savefolder.joinpath('UFLUX_multiyear_diff_LUCC_sum.csv'))
        df_lucc_only.to_csv(savefolder.joinpath('UFLUX_multiyear_diff_LUCC_only.csv'))


        if ncu_avg['NEP'].shape[0] * ncu_avg['NEP'].shape[1] > 1000 * 1000:
            dd_avg_moran, dd_avg_lisa = moran_list(ncu_avg['NEP'].coarsen(latitude = 4, longitude = 4, boundary = 'trim').mean())
            dd_lucc_moran, dd_lucc_lisa = moran_list(ncd['NEP'].coarsen(latitude = 4, longitude = 4, boundary = 'trim').mean())
        else:
            dd_avg_moran, dd_avg_lisa = moran_list(ncu_avg['NEP'])
            if city_name == 'Accra':
                dd_lucc_moran = {'Moran_I': None, 'p_value': None, 'N_valid_pixels': None}
                dd_lucc_lisa = {
                    'quadrant_counts': {'NE_HH': None, 'NW_HH': None, 'SE_HH': None, 'SW_HH': None, 'NE_LL': None, 'NW_LL': None, 'SE_LL': None, 'SW_LL': None},
                    'quadrant_percentages': {'NE_HH': None, 'NW_HH': None, 'SE_HH': None, 'SW_HH': None, 'NE_LL': None, 'NW_LL': None, 'SE_LL': None, 'SW_LL': None},
                    'centre_vs_periphery_percent': {'centre_HH_%': None, 'periphery_HH_%': None, 'centre_LL_%': None, 'periphery_LL_%': None}
                }
            else:
                dd_lucc_moran, dd_lucc_lisa = moran_list(ncd['NEP'])

        joblib.dump(dd_avg_moran, savefolder.joinpath('UFLUX_multiyear_average_Moran.joblib'))
        joblib.dump(dd_avg_lisa, savefolder.joinpath('UFLUX_multiyear_average_local_Moran.joblib'))
        joblib.dump(dd_lucc_moran, savefolder.joinpath('UFLUX_multiyear_lucc_Moran.joblib'))
        joblib.dump(dd_lucc_lisa, savefolder.joinpath('UFLUX_multiyear_lucc_local_Moran.joblib'))



        dfd = ncd.drop_vars('NEP_dir').to_dataframe().dropna()
        if dfd.empty:
            dd_out = {}

        else:
            dd_out = {
                'data': dfd,
            }


            X = dfd[['EVI', 'T2m', 'Tsoil1', 'Swvl1', 'SWIN', 'LWIN', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE', 'WUE']]
            y = dfd['NEP']

            dd_out['NEP_linear'] = run_linear(X, y)
            dd_out['NEP_xgboost'] = run_xgboost(X, y)

            X = dfd[['EVI', 'T2m', 'Tsoil1', 'Swvl1', 'SWIN', 'LWIN', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE', 'WUE']]
            y = dfd['Rw']

            dd_out['Rw_linear'] = run_linear(X, y)
            dd_out['Rw_xgboost'] = run_xgboost(X, y)

        joblib.dump(dd_out, savefolder.joinpath('UFLUX_multiyear_lucc_ML_casual.joblib'))


def run_drivers():
    'v3 cell 78: Ridge/VIF and cross-city model summaries; exports original in-sample Ridge metrics.'
    df_path = []
    for p in root_proj.joinpath('4_analysis').rglob('UFLUX_multiyear_lucc_ML_casual.joblib'):
        df_path.append([p.parent.stem, p])
    df_path = pd.DataFrame(df_path, columns = ['CITY', 'PATH']).set_index('CITY')


    df_perf = []; df_nep_lin_imp = []; df_nep_xgb_imp = []; df_rw_lin_imp = []; df_rw_xgb_imp = []
    df_nep_rdg_beta_std = []; df_rw_rdg_beta_std = []; df_nep_rdg_beta_actual = []; df_rw_rdg_beta_actual = [];
    df_nep_rdg_vif = []; df_rw_rdg_vif = [];
    for city_name in tqdm(df_path.index):
        if city_name in ['Dakar', 'Accra']: continue
        p_lucc_ml = df_path.loc[city_name, 'PATH']
        dd_ml = joblib.load(p_lucc_ml)

        dd_nep_rdg = get_ridge_reg(dd_ml, 'NEP', X_names = ['EVI', 'T2m', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE', 'WUE'])
        dd_rw_rdg = get_ridge_reg(dd_ml, 'Rw', X_names = ['EVI', 'T2m', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE', 'WUE'])

        dd_nep_lin = dd_ml['NEP_linear']
        dd_nep_xgb = dd_ml['NEP_xgboost']
        dd_rw_lin = dd_ml['Rw_linear']
        dd_rw_xgb = dd_ml['Rw_xgboost']

        df_perf.append([
            city_name,
            dd_nep_lin['r2'], dd_nep_lin['rmse'],
            dd_nep_xgb['r2'], dd_nep_xgb['rmse'],
            dd_rw_lin['r2'], dd_rw_lin['rmse'],
            dd_rw_xgb['r2'], dd_rw_xgb['rmse'],
            dd_nep_rdg['r2'], dd_nep_rdg['rmse'],
            dd_rw_rdg['r2'], dd_rw_rdg['rmse'],
            dd_nep_rdg['alpha'], dd_rw_rdg['alpha']
        ])

        df_nep_lin_imp.append(dd_nep_lin['importance'].rename(city_name).to_frame().T)
        df_nep_xgb_imp.append(
            pd.Series(
                dd_nep_xgb['model'].feature_importances_,
                index = ['EVI', 'T2m', 'Tsoil1', 'Swvl1', 'SWIN', 'LWIN', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE', 'WUE']
            ).rename(city_name).to_frame().T
        )
        df_rw_lin_imp.append(dd_rw_lin['importance'].rename(city_name).to_frame().T)
        df_rw_xgb_imp.append(
            pd.Series(
                dd_rw_xgb['model'].feature_importances_,
                index = ['EVI', 'T2m', 'Tsoil1', 'Swvl1', 'SWIN', 'LWIN', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE', 'WUE']
            ).rename(city_name).to_frame().T
        )

        df_nep_rdg_beta_std.append(dd_nep_rdg['coef_vif']['beta_std'].rename(city_name).to_frame().T)
        df_rw_rdg_beta_std.append(dd_rw_rdg['coef_vif']['beta_std'].rename(city_name).to_frame().T)
        df_nep_rdg_beta_actual.append(dd_nep_rdg['coef_vif']['beta_actual'].rename(city_name).to_frame().T)
        df_rw_rdg_beta_actual.append(dd_rw_rdg['coef_vif']['beta_actual'].rename(city_name).to_frame().T)
        df_nep_rdg_vif.append(dd_nep_rdg['coef_vif']['VIF'].rename(city_name).to_frame().T)
        df_rw_rdg_vif.append(dd_rw_rdg['coef_vif']['VIF'].rename(city_name).to_frame().T)

    df_perf = pd.DataFrame(df_perf, columns = [
        'city',
        'NEP_linear_R2', 'NEP_linear_RMSE', 'NEP_xgboost_R2', 'NEP_xgboost_RMSE',
        'Rw_linear_R2', 'Rw_linear_RMSE', 'Rw_xgboost_R2', 'Rw_xgboost_RMSE',
        'NEP_ridge_R2', 'NEP_ridge_RMSE', 'Rw_ridge_R2', 'Rw_ridge_RMSE',
        'NEP_ridge_alpha', 'Rw_ridge_alpha'
    ]).set_index('city')

    df_nep_lin_imp = pd.concat(df_nep_lin_imp)
    df_nep_xgb_imp = pd.concat(df_nep_xgb_imp)
    df_rw_lin_imp = pd.concat(df_rw_lin_imp)
    df_rw_xgb_imp = pd.concat(df_rw_xgb_imp)

    df_nep_rdg_beta_std = pd.concat(df_nep_rdg_beta_std)
    df_nep_rdg_beta_actual = pd.concat(df_nep_rdg_beta_actual)
    df_nep_rdg_vif = pd.concat(df_nep_rdg_vif)
    df_rw_rdg_beta_std = pd.concat(df_rw_rdg_beta_std)
    df_rw_rdg_beta_actual = pd.concat(df_rw_rdg_beta_actual)
    df_rw_rdg_vif = pd.concat(df_rw_rdg_vif)
    export_dir = root_proj / '5_supp' / 'analysis_exports'
    export_dir.mkdir(parents=True, exist_ok=True)
    for name, value in list(locals().items()):
        if name.startswith('df_') and isinstance(value, pd.DataFrame):
            value.to_csv(export_dir / (name + '.csv'))


def run_tree():
    'v3 cell 148: original tiled area-weighted analysis; original MIN_YEARS=8 retained.'
    from pathlib import Path
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import xarray as xr
    from pyproj import Geod
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds

    INPUT = root_proj / "4_analysis/spatial_EVI_DynamicWorld_UFLUXv1"
    OUTPUT = root_proj / "5_supp/tree_audit/tree_baseline_fast.csv"

    TILE = 256
    MIN_YEARS = 8
    CITIES = None                 # Set to ["Lagos", "London"] for a preliminary check; use None for all cities.
    FORCE_RECALCULATE = False     # Set True to recalculate existing results.

    def calculate_city(path):
        city = path.name.removesuffix("_uflux.nc")
        boundary = root_proj / "2_geojsons" / f"{city}.geojson"

        geo = gpd.read_file(boundary).to_crs("EPSG:4326")
        shapes = [f["geometry"] for f in geo.geometry.__geo_interface__["features"]]
        geod = Geod(ellps="WGS84")

        area_by_class = np.zeros(9)
        positive_by_class = np.zeros(9)
        signed_by_class = np.zeros(9)

        with xr.open_dataset(path) as ds:
            ds = ds[["NEP", "LC"]]
            years = pd.DatetimeIndex(ds.time.values).year
            selected = np.flatnonzero((years >= 2016) & (years <= 2025))

            if len(selected) != 10 or len(set(years[selected])) != 10:
                raise ValueError("Exactly one annual grid is required for each year from 2016 to 2025.")

            ds = ds.isel(time=selected)
            lat = ds.latitude.values
            lon = ds.longitude.values
            dy = abs(float(np.median(np.diff(lat))))
            dx = abs(float(np.median(np.diff(lon))))

            if (not np.allclose(abs(np.diff(lat)), dy, rtol=1e-3) or
                not np.allclose(abs(np.diff(lon)), dx, rtol=1e-3)):
                raise ValueError("The latitude/longitude grid is irregular.")

            row_area = np.array([
                abs(geod.polygon_area_perimeter(
                    [-dx/2, dx/2, dx/2, -dx/2],
                    [y-dy/2, y-dy/2, y+dy/2, y+dy/2]
                )[0])
                for y in lat
            ])

            for i in range(0, len(lat), TILE):
                for j in range(0, len(lon), TILE):
                    ys = lat[i:i+TILE]
                    xs = lon[j:j+TILE]

                    transform = from_bounds(
                        xs.min()-dx/2, ys.min()-dy/2,
                        xs.max()+dx/2, ys.max()+dy/2,
                        len(xs), len(ys)
                    )
                    inside = geometry_mask(
                        shapes,
                        out_shape=(len(ys), len(xs)),
                        transform=transform,
                        invert=True
                    )
                    if ys[0] < ys[-1]:
                        inside = inside[::-1]
                    if not inside.any():
                        continue

                    part = ds.isel(
                        latitude=slice(i, i+TILE),
                        longitude=slice(j, j+TILE)
                    )
                    nep = part.NEP.values
                    lc = part.LC.values

                    nep_ok = np.isfinite(nep)
                    lc_ok = (
                        np.isfinite(lc) &
                        (lc >= 0) & (lc <= 8) &
                        (lc == np.floor(np.where(np.isfinite(lc), lc, -1)))
                    )
                    valid = (
                        inside &
                        (nep_ok.sum(axis=0) >= MIN_YEARS) &
                        (lc_ok.sum(axis=0) >= MIN_YEARS)
                    )
                    if not valid.any():
                        continue

                    counts = np.stack([(lc == k).sum(axis=0) for k in range(9)])
                    mode = counts.argmax(axis=0)
                    mean_nep = np.nanmean(
                        np.where(nep_ok, nep, np.nan), axis=0
                    )
                    pixel_area = np.broadcast_to(
                        row_area[i:i+len(ys), None], mode.shape
                    )

                    codes = mode[valid].astype(int)
                    weights = pixel_area[valid]
                    flux = mean_nep[valid]

                    area_by_class += np.bincount(
                        codes, weights=weights, minlength=9
                    )
                    positive_by_class += np.bincount(
                        codes, weights=np.maximum(flux, 0) * weights, minlength=9
                    )
                    signed_by_class += np.bincount(
                        codes, weights=flux * weights, minlength=9
                    )

        total_area = area_by_class.sum()
        total_positive = positive_by_class.sum()
        vegetation_area = area_by_class[1:6].sum()
        vegetation_positive = positive_by_class[1:6].sum()

        return {
            "city": city,
            "valid_area_km2": total_area / 1e6,
            "tree_area_share_all_valid":
                area_by_class[1] / total_area if total_area else np.nan,
            "tree_positive_share_all_valid":
                positive_by_class[1] / total_positive if total_positive else np.nan,
            "tree_area_share_vegetation":
                area_by_class[1] / vegetation_area if vegetation_area else np.nan,
            "tree_positive_share_vegetation":
                positive_by_class[1] / vegetation_positive
                if vegetation_positive else np.nan,
            "city_nep_gC_m2_day":
                signed_by_class.sum() / total_area if total_area else np.nan,
            "city_signed_MgC_yr": signed_by_class.sum() * 365 / 1e6,
            "all_positive_MgC_yr": total_positive * 365 / 1e6,
            **{
                f"class_{k}_area_km2": area_by_class[k] / 1e6
                for k in range(9)
            },
            **{
                f"class_{k}_positive_MgC_yr":
                    positive_by_class[k] * 365 / 1e6
                for k in range(9)
            },
            **{
                f"class_{k}_signed_MgC_yr":
                    signed_by_class[k] * 365 / 1e6
                for k in range(9)
            },
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    paths = sorted(INPUT.glob("*_uflux.nc"))

    if CITIES is not None:
        paths = [
            p for p in paths
            if p.name.removesuffix("_uflux.nc") in CITIES
        ]

    for path in paths:
        city = path.name.removesuffix("_uflux.nc")

        if (not FORCE_RECALCULATE and
            not results.empty and
            city in set(results["city"])):
            print(f"Skipping completed city: {city}", flush=True)
            continue

        try:
            row = calculate_city(path)
            if not results.empty:
                results = results[results["city"] != city]
            results = pd.concat(
                [results, pd.DataFrame([row])], ignore_index=True
            )
            results.to_csv(OUTPUT, index=False)

            print(
                f"{city}: tree area share "
                f"{row['tree_area_share_all_valid']:.1%}; "
                f"positive NEP share "
                f"{row['tree_positive_share_all_valid']:.1%}",
                flush=True
            )
        except Exception as exc:
            raise RuntimeError(f"{city}: analysis failed") from exc

    print(f"\nResults saved to: {OUTPUT}")
    print(results[[
        "city",
        "tree_area_share_all_valid",
        "tree_positive_share_all_valid",
        "city_nep_gC_m2_day"
    ]])


def run_robustness():
    'v3 cell 150: original tiled area-weighted analysis; original MIN_YEARS=8 retained.'
    from pathlib import Path
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import xarray as xr
    from pyproj import Geod
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds

    INPUT = root_proj / "4_analysis/spatial_EVI_DynamicWorld_UFLUXv1"
    OUTPUT = root_proj / "5_supp/tree_audit"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    CITIES = None
    TILE = 256
    MIN_YEARS = 8
    FORCE_RECALCULATE = False

    SCENARIOS = (
        "baseline", "stable_landcover", "vegetated_only",
        "built_only", "non_built", "evi_gt_0p05",
        "period_2016_2020", "period_2021_2025",
        "alternative_minus_NEE", "GPP_low_RECO_high",
        "GPP_high_RECO_low"
    )

    def time_mean(a):
        count = np.isfinite(a).sum(axis=0)
        result = np.full(a.shape[1:], np.nan)
        np.divide(
            np.nansum(a, axis=0), count,
            out=result, where=count > 0
        )
        return result

    def modal_lc(a):
        valid = np.isfinite(a) & (a >= 0) & (a <= 8)
        counts = np.stack([(a == k).sum(axis=0) for k in range(9)])
        n = valid.sum(axis=0)
        return counts.argmax(axis=0), counts.max(axis=0) / np.maximum(n, 1), n

    def analyse_city(path):
        city = path.name.removesuffix("_uflux.nc")
        boundary = root_proj / "2_geojsons" / f"{city}.geojson"
        geo = gpd.read_file(boundary).to_crs("EPSG:4326")
        shapes = [
            f["geometry"]
            for f in geo.geometry.__geo_interface__["features"]
        ]

        sums = {name: np.zeros(3) for name in SCENARIOS}
        annual = np.zeros((10, 5))
        diagnostics = np.zeros(4)
        direct_diagnostics = np.zeros(3)
        geod = Geod(ellps="WGS84")

        with xr.open_dataset(path) as ds:
            years = pd.DatetimeIndex(ds.time.values).year
            selected = np.flatnonzero((years >= 2016) & (years <= 2025))
            if len(selected) != 10 or len(set(years[selected])) != 10:
                raise ValueError("Exactly one annual grid is required for each year from 2016 to 2025.")

            ds = ds.isel(time=selected)
            required = {"NEP", "LC", "GPP", "RECO"}
            if not required.issubset(ds.data_vars):
                raise ValueError(f"Missing variables: {required - set(ds.data_vars)}")

            has_direct = "NEP_dir" in ds
            has_evi = "EVI" in ds
            variables = ["NEP", "LC", "GPP", "RECO"]
            if has_direct:
                variables.append("NEP_dir")
            if has_evi:
                variables.append("EVI")

            lat = ds.latitude.values
            lon = ds.longitude.values
            dy = abs(float(np.median(np.diff(lat))))
            dx = abs(float(np.median(np.diff(lon))))

            if (
                not np.allclose(abs(np.diff(lat)), dy, rtol=1e-3)
                or not np.allclose(abs(np.diff(lon)), dx, rtol=1e-3)
            ):
                raise ValueError("The latitude/longitude grid is irregular.")

            row_area = np.array([
                abs(geod.polygon_area_perimeter(
                    [-dx/2, dx/2, dx/2, -dx/2],
                    [y-dy/2, y-dy/2, y+dy/2, y+dy/2]
                )[0])
                for y in lat
            ])

            for i in range(0, len(lat), TILE):
                for j in range(0, len(lon), TILE):
                    ys = lat[i:i+TILE]
                    xs = lon[j:j+TILE]

                    transform = from_bounds(
                        xs.min()-dx/2, ys.min()-dy/2,
                        xs.max()+dx/2, ys.max()+dy/2,
                        len(xs), len(ys)
                    )
                    inside = geometry_mask(
                        shapes,
                        out_shape=(len(ys), len(xs)),
                        transform=transform,
                        invert=True
                    )
                    if ys[0] < ys[-1]:
                        inside = inside[::-1]
                    if not inside.any():
                        continue

                    part = ds[variables].isel(
                        latitude=slice(i, i+TILE),
                        longitude=slice(j, j+TILE)
                    ).load()

                    nep = part.NEP.values
                    lc = part.LC.values
                    gpp = part.GPP.values
                    reco = part.RECO.values
                    area = np.broadcast_to(
                        row_area[i:i+len(ys), None], inside.shape
                    )
                    diagnostics[0] += area[inside].sum()

                    mode, agreement, lc_years = modal_lc(lc)
                    valid = (
                        inside
                        & (np.isfinite(nep).sum(axis=0) >= MIN_YEARS)
                        & (lc_years >= MIN_YEARS)
                    )
                    stable = valid & (agreement >= 0.8)

                    n = time_mean(nep)
                    g = time_mean(gpp)
                    r = time_mean(reco)

                    diagnostics[1] += area[valid].sum()
                    diagnostics[2] += area[stable].sum()

                    common_gpp = valid & np.isfinite(n) & np.isfinite(g) & np.isfinite(r)
                    diagnostics[3] += (
                        abs(n[common_gpp] - (g-r)[common_gpp])
                        * area[common_gpp]
                    ).sum()

                    def record(label, flux, mask):
                        ok = mask & np.isfinite(flux)
                        if not ok.any():
                            return
                        a = area[ok]
                        f = flux[ok]
                        sums[label] += [
                            a.sum(),
                            (f * a).sum(),
                            (np.maximum(f, 0) * a).sum()
                        ]

                    record("baseline", n, valid)
                    record("stable_landcover", n, stable)
                    record(
                        "vegetated_only", n,
                        valid & np.isin(mode, [1, 2, 3, 4, 5])
                    )
                    record("built_only", n, valid & (mode == 6))
                    record("non_built", n, valid & (mode != 6))

                    if has_evi:
                        record(
                            "evi_gt_0p05", n,
                            valid & (time_mean(part.EVI.values) > 0.05)
                        )

                    for label, years_slice in [
                        ("period_2016_2020", slice(0, 5)),
                        ("period_2021_2025", slice(5, 10))
                    ]:
                        _, _, period_lc_years = modal_lc(lc[years_slice])
                        period_valid = (
                            inside
                            & (np.isfinite(nep[years_slice]).sum(axis=0) >= 4)
                            & (period_lc_years >= 4)
                        )
                        record(
                            label, time_mean(nep[years_slice]), period_valid
                        )

                    if has_direct:
                        direct = time_mean(part.NEP_dir.values)
                        common = valid & np.isfinite(n) & np.isfinite(direct)
                        record("alternative_minus_NEE", direct, common)
                        a = area[common]
                        direct_diagnostics += [
                            a.sum(),
                            (abs(n[common] - direct[common]) * a).sum(),
                            (
                                (np.sign(n[common]) != np.sign(direct[common]))
                                * a
                            ).sum()
                        ]

                    record(
                        "GPP_low_RECO_high",
                        0.9 * g - 1.1 * r,
                        common_gpp
                    )
                    record(
                        "GPP_high_RECO_low",
                        1.1 * g - 0.9 * r,
                        common_gpp
                    )

                    for t in range(10):
                        m = (
                            inside & np.isfinite(nep[t])
                            & np.isfinite(lc[t])
                            & (lc[t] >= 0) & (lc[t] <= 8)
                        )
                        annual[t, 0] += area[m].sum()
                        fields = [
                            nep[t], gpp[t], reco[t],
                            part.NEP_dir.values[t] if has_direct else None
                        ]
                        for k, field in enumerate(fields, start=1):
                            if field is not None:
                                ok = m & np.isfinite(field)
                                annual[t, k] += (area[ok] * field[ok]).sum()

        scenario_rows = []
        for label, values in sums.items():
            a, signed, positive = values
            if a == 0:
                continue
            scenario_rows.append({
                "city": city,
                "scenario": label,
                "valid_area_km2": a / 1e6,
                "city_nep_gC_m2_day": signed / a,
                "city_signed_MgC_yr": signed * 365 / 1e6,
                "positive_MgC_yr": positive * 365 / 1e6
            })

        all_area, valid_area, stable_area, identity_gap = diagnostics
        common_area, direct_gap, sign_disagreement = direct_diagnostics
        diagnostic_row = {
            "city": city,
            "area_valid_fraction_of_boundary_grid":
                valid_area / all_area if all_area else np.nan,
            "area_stable_fraction_of_valid":
                stable_area / valid_area if valid_area else np.nan,
            "nep_vs_gpp_minus_reco_mean_absolute_gap":
                identity_gap / valid_area if valid_area else np.nan,
            "nep_vs_minus_nee_mean_absolute_gap":
                direct_gap / common_area if common_area else np.nan,
            "nep_vs_minus_nee_sign_disagreement_area_fraction":
                sign_disagreement / common_area if common_area else np.nan
        }

        annual_rows = []
        for t, year in enumerate(years[selected]):
            a = annual[t, 0]
            annual_rows.append({
                "city": city,
                "year": int(year),
                "valid_area_fraction_of_boundary_grid":
                    a / all_area if all_area else np.nan,
                "nep_gC_m2_day": annual[t, 1] / a if a else np.nan,
                "gpp_gC_m2_day": annual[t, 2] / a if a else np.nan,
                "reco_gC_m2_day": annual[t, 3] / a if a else np.nan,
                "minus_nee_gC_m2_day":
                    annual[t, 4] / a if a and has_direct else np.nan
            })

        return scenario_rows, diagnostic_row, annual_rows

    files = {
        "scenarios": OUTPUT / "city_scenarios_fast.csv",
        "diagnostics": OUTPUT / "flux_diagnostics_fast.csv",
        "annual": OUTPUT / "annual_city_flux_fast.csv"
    }
    tables = {
        key: pd.read_csv(path) if path.exists() else pd.DataFrame()
        for key, path in files.items()
    }

    paths = sorted(INPUT.glob("*_uflux.nc"))
    if CITIES is not None:
        paths = [
            p for p in paths
            if p.name.removesuffix("_uflux.nc") in CITIES
        ]

    for path in paths:
        city = path.name.removesuffix("_uflux.nc")

        if (
            not FORCE_RECALCULATE
            and not tables["diagnostics"].empty
            and city in set(tables["diagnostics"]["city"])
        ):
            print(f"Skipping completed city: {city}", flush=True)
            continue

        try:
            scenarios, diagnostic, annual_rows = analyse_city(path)
            updates = {
                "scenarios": scenarios,
                "diagnostics": [diagnostic],
                "annual": annual_rows
            }

            for key, rows in updates.items():
                old = tables[key]
                if not old.empty:
                    old = old[old["city"] != city]
                tables[key] = pd.concat(
                    [old, pd.DataFrame(rows)], ignore_index=True
                )
                tables[key].to_csv(files[key], index=False)

            print(
                f"{city}: baseline NEP = "
                f"{scenarios[0]['city_nep_gC_m2_day']:.3f} "
                f"g C m⁻² d⁻¹; valid area coverage = "
                f"{diagnostic['area_valid_fraction_of_boundary_grid']:.1%}",
                flush=True
            )
        except Exception as exc:
            raise RuntimeError(f"{city}: analysis failed") from exc

    print("\nOutput files:")
    for path in files.values():
        print(path)

    print(tables["diagnostics"])


def run_ablation():
    'v3 cell 146: original exogenous efficiency ablation; formula/units and sampling limitations are documented above.'
    print(">>> Searching for city joblib files...")

    joblib_files = list(root_proj.joinpath('4_analysis').rglob('UFLUX_multiyear_lucc_ML_casual.joblib'))

    df_path = []

    for p in joblib_files:
        df_path.append([p.parent.stem, p])

    df_path = pd.DataFrame(df_path, columns=['CITY', 'PATH']).set_index('CITY')

    print(f"Found {len(df_path)} city model files.")

    def calc_theoretical_lue_iwue(T_air_C, P_atm_hPa, VPD_hPa, Ca_ppm=415.0):
        """
        Calculate efficiencies using the stomatal optimality formulation attributed to Wang et al. (2017):
        - Theoretical light-use efficiency LUE_photo (g C MJ-1 APAR)
        - Theoretical intrinsic water-use efficiency IWUE (g C kg-1 H2O)
        Driven only by external environmental inputs (T2m, PA, VPD, CO2), without GPP in the numerator.
        """
        Tk = T_air_C + 273.15
        P_atm_Pa = np.clip(P_atm_hPa * 100.0, 50000.0, 110000.0)  # hPa -> Pa
        VPD_kPa = np.clip(VPD_hPa * 0.1, 0.01, 10.0)              # hPa -> kPa
        R_gas = 8.314  # J mol-1 K-1

        Ca = Ca_ppm * 1e-6 * P_atm_Pa

        Gamma_star = 42.75 * np.exp((37830.0 / R_gas) * (1.0 / 298.15 - 1.0 / Tk))

        Kc = 404.9 * np.exp((79430.0 / R_gas) * (1.0 / 298.15 - 1.0 / Tk))
        Ko = 278.4 * np.exp((36380.0 / R_gas) * (1.0 / 298.15 - 1.0 / Tk))
        O2_partial = 0.209 * P_atm_Pa
        Km = Kc * (1.0 + O2_partial / Ko)

        eta_star = np.exp(-0.024 * (T_air_C - 25.0))

        beta = 146.0
        xi = np.sqrt((beta * (Km + Gamma_star)) / (1.6 * eta_star))

        chi = Gamma_star / Ca + (1.0 - Gamma_star / Ca) * (xi / (xi + np.sqrt(VPD_kPa)))
        chi = np.clip(chi, 0.05, 0.95)

        iwue_umol_mol = (Ca / P_atm_Pa * 1e6) * (1.0 - chi) / 1.6
        iwue_g_kg = np.clip(iwue_umol_mol * (12.011 / 18.015) * 1e-3, 0.0, 10.0)

        m_prime = (Ca * (1.0 - chi) - Gamma_star) / (Ca * (1.0 - chi) + 2.0 * Gamma_star)
        m_prime = np.clip(m_prime, 0.0, 1.0)
        phi0 = 0.08  # Maximum intrinsic quantum yield (mol C / mol absorbed photons)
        lue_theoretical = phi0 * m_prime * (12.011 / 0.217)
        lue_theoretical = np.clip(lue_theoretical, 0.0, 5.0)

        return lue_theoretical, iwue_g_kg

    def run_ridge_pipeline(df, target_name, x_cols):
        X = df[x_cols].copy().fillna(0)
        y = df[target_name].copy().fillna(0)

        if len(X) > 10000:
            idx = np.random.choice(len(X), size=10000, replace=False)
            X_sample = X.iloc[idx]
            y_sample = y.iloc[idx]
        else:
            X_sample = X
            y_sample = y

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)

        alphas = np.logspace(-2, 3, 6)
        ridge_cv = RidgeCV(alphas=alphas, cv=3)
        ridge_cv.fit(X_scaled, y_sample)

        ridge_model = Ridge(alpha=ridge_cv.alpha_, solver='lsqr')
        ridge_model.fit(X_scaled, y_sample)

        beta_std = pd.Series(ridge_model.coef_, index=x_cols)
        norm_imp = beta_std.abs() / (beta_std.abs().sum() + 1e-12)
        return norm_imp

    def run_xgboost_pipeline(df, target_name, x_cols):
        X = df[x_cols].copy().fillna(0)
        y = df[target_name].copy().fillna(0)

        if len(X) > 10000:
            idx = np.random.choice(len(X), size=10000, replace=False)
            X_sample = X.iloc[idx]
            y_sample = y.iloc[idx]
        else:
            X_sample = X
            y_sample = y

        model_xgb = XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model_xgb.fit(X_sample, y_sample)
        imp = pd.Series(model_xgb.feature_importances_, index=x_cols)
        if imp.sum() > 0:
            imp = imp / imp.sum()
        return imp

    print(">>> Running Full Exogenous Decoupling Ablation across 70 cities...")

    cities_to_run = [c for c in df_path.index if c not in ['Dakar', 'Accra']]

    feat_orig = ['EVI', 'T2m', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE', 'WUE']

    feat_mech = ['EVI', 'T2m', 'PA', 'P', 'VPD', 'Rnet', 'LUCC', 'LUE_photo', 'IWUE']

    res_xgb_orig, res_xgb_mech = [], []

    res_rdg_orig, res_rdg_mech = [], []

    for city_name in tqdm(cities_to_run):
        try:
            p_lucc_ml = df_path.loc[city_name, 'PATH']
            dd_ml = joblib.load(p_lucc_ml)
            if 'data' not in dd_ml or dd_ml['data'].empty:
                continue
            df_city = dd_ml['data'].copy()

            lue_photo, iwue_photo = calc_theoretical_lue_iwue(
                T_air_C=df_city['T2m'].values,
                P_atm_hPa=df_city['PA'].values,
                VPD_hPa=df_city['VPD'].values,
                Ca_ppm=415.0
            )
            df_city['LUE_photo'] = lue_photo
            df_city['IWUE'] = iwue_photo

            res_xgb_orig.append(run_xgboost_pipeline(df_city, 'NEP', feat_orig).rename(city_name))
            res_rdg_orig.append(run_ridge_pipeline(df_city, 'NEP', feat_orig).rename(city_name))

            res_xgb_mech.append(run_xgboost_pipeline(df_city, 'NEP', feat_mech).rename(city_name))
            res_rdg_mech.append(run_ridge_pipeline(df_city, 'NEP', feat_mech).rename(city_name))

        except Exception as e:
            print(f"Skipping {city_name}: {e}")
            continue

    df_xgb_orig = pd.concat(res_xgb_orig, axis=1).T

    df_xgb_mech = pd.concat(res_xgb_mech, axis=1).T

    df_rdg_orig = pd.concat(res_rdg_orig, axis=1).T

    df_rdg_mech = pd.concat(res_rdg_mech, axis=1).T

    print("\n>>> Full Ablation Analysis Completed!")

    df_xgb_orig_p = df_xgb_orig.rename(columns={'WUE': 'Water Efficiency', 'LUE': 'Light Efficiency'})

    df_xgb_mech_p = df_xgb_mech.rename(columns={'IWUE': 'Water Efficiency', 'LUE_photo': 'Light Efficiency'})

    df_rdg_orig_p = df_rdg_orig.rename(columns={'WUE': 'Water Efficiency', 'LUE': 'Light Efficiency'})

    df_rdg_mech_p = df_rdg_mech.rename(columns={'IWUE': 'Water Efficiency', 'LUE_photo': 'Light Efficiency'})

    plot_order = ['Water Efficiency', 'Light Efficiency', 'EVI', 'VPD', 'Rnet', 'T2m', 'P', 'PA', 'LUCC']

    summary_xgb = pd.DataFrame({
        'Feature': plot_order,
        'Orig_mean': [df_xgb_orig_p[f].mean() for f in plot_order],
        'Orig_std': [df_xgb_orig_p[f].std() for f in plot_order],
        'Mech_mean': [df_xgb_mech_p[f].mean() for f in plot_order],
        'Mech_std': [df_xgb_mech_p[f].std() for f in plot_order],
    })

    summary_rdg = pd.DataFrame({
        'Feature': plot_order,
        'Orig_mean': [df_rdg_orig_p[f].mean() for f in plot_order],
        'Orig_std': [df_rdg_orig_p[f].std() for f in plot_order],
        'Mech_mean': [df_rdg_mech_p[f].mean() for f in plot_order],
        'Mech_std': [df_rdg_mech_p[f].std() for f in plot_order],
    })
    export_dir = root_proj / '5_supp' / 'exogenous_ablation'
    export_dir.mkdir(parents=True, exist_ok=True)
    for name in ['df_xgb_orig', 'df_xgb_mech', 'df_rdg_orig', 'df_rdg_mech', 'summary_xgb', 'summary_rdg']:
        locals()[name].to_csv(export_dir / (name + '.csv'))


def run_spatial():
    'v3 cell 142: area-normalised baseline/delta LISA, original 200 m / 99 permutations.'
    SUFFIX = '_EVI_DynamicWorld_UFLUXv1'
    PERMUTATIONS = 99
    TARGET_RESOLUTION_M = 200
    MAX_ANALYSIS_CELLS = 30_000
    def _metric_coordinates(da):
        """Convert the latitude/longitude grid to local UTM coordinates in metres."""
        lat, lon = np.meshgrid(
            da.latitude.values,
            da.longitude.values,
            indexing="ij",
        )

        lat0 = float(np.nanmedian(lat))
        lon0 = float(np.nanmedian(lon))

        utm_zone = int(np.floor((lon0 + 180) / 6) + 1)
        utm_zone = int(np.clip(utm_zone, 1, 60))
        epsg = (32600 if lat0 >= 0 else 32700) + utm_zone

        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{epsg}",
            always_xy=True,
        )

        x, y = transformer.transform(lon, lat)
        return x, y

    def _coarsening_factors(da, target_m):
        """Determine coarsening factors from the north-south and east-west pixel dimensions."""
        x, y = _metric_coordinates(da)

        if da.sizes["latitude"] < 2 or da.sizes["longitude"] < 2:
            raise ValueError("The grid must have at least two pixels along each spatial dimension.")

        dy = float(np.nanmedian(np.hypot(
            np.diff(x, axis=0),
            np.diff(y, axis=0),
        )))

        dx = float(np.nanmedian(np.hypot(
            np.diff(x, axis=1),
            np.diff(y, axis=1),
        )))

        if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
            raise ValueError(f"Cannot determine pixel dimensions: dx={dx}, dy={dy}")

        factor_y = max(1, int(round(target_m / dy)))
        factor_x = max(1, int(round(target_m / dx)))

        return factor_y, factor_x, dy, dx

    def _grid_weights(valid):
        """
        Build four-neighbour spatial weights for valid pixels only.
        Avoid constructing lat2W weights for the entire original grid.
        """
        ny, nx = valid.shape
        flat_valid = valid.ravel()
        ids = np.flatnonzero(flat_valid)

        neighbours = {}

        for raw_id in ids:
            i = int(raw_id)
            row, col = divmod(i, nx)

            candidates = []
            if row > 0:
                candidates.append(i - nx)
            if row < ny - 1:
                candidates.append(i + nx)
            if col > 0:
                candidates.append(i - 1)
            if col < nx - 1:
                candidates.append(i + 1)

            neighbours[i] = [
                j for j in candidates if flat_valid[j]
            ]

        weights = {
            i: [1.0] * len(js)
            for i, js in neighbours.items()
        }

        w = libpysal.weights.W(
            neighbours,
            weights,
            id_order=[int(i) for i in ids],
            silence_warnings=True,
        )

        return ids, w

    def _lisa_cluster_map(values, valid, ids, w, permutations, seed=42):
        """Return 0 for nonsignificant pixels, 1 for HH and 3 for LL; retain other quadrant codes."""
        lisa = Moran_Local(
            values.ravel()[ids],
            w,
            permutations=permutations,
            seed=seed,
        )

        cluster_flat = np.zeros(values.size, dtype=np.int8)

        cluster_flat[ids] = np.where(
            lisa.p_sim < 0.05,
            lisa.q,
            0,
        )

        return cluster_flat.reshape(values.shape)

    def analyse_city_lisa(city):
        d = load_city_uflux(
            city,
            SUFFIX,
            period1=["2016", "2020"],
            period2=["2021", "2025"],
        )

        baseline = d["uflux_period0_avg"]["NEP"]
        delta = d["uflux_periods_diff"]["NEP"]

        factor_y, factor_x, original_dy, original_dx = (
            _coarsening_factors(baseline, TARGET_RESOLUTION_M)
        )

        baseline = baseline.coarsen(
            latitude=factor_y,
            longitude=factor_x,
            boundary="trim",
        ).mean()

        delta = delta.coarsen(
            latitude=factor_y,
            longitude=factor_x,
            boundary="trim",
        ).mean()

        base_values = np.asarray(baseline.values, dtype=float)
        delta_values = np.asarray(delta.values, dtype=float)

        valid = np.isfinite(base_values) & np.isfinite(delta_values)
        n_valid = int(valid.sum())

        if n_valid < 50:
            raise ValueError(f"{city}: only {n_valid} pixels are valid in both datasets")

        if n_valid > MAX_ANALYSIS_CELLS:
            raise ValueError(
                f"{city}: {n_valid} valid pixels remain after coarsening, exceeding "
                f"{MAX_ANALYSIS_CELLS}; increase the common "
                "TARGET_RESOLUTION_M for all cities, select a new output directory, and rerun all cities"
            )

        x, y = _metric_coordinates(baseline)

        xc = float(np.median(x[valid]))
        yc = float(np.median(y[valid]))

        distance = np.hypot(x - xc, y - yc)
        cutoff = float(np.percentile(distance[valid], 30))

        centre = valid & (distance <= cutoff)
        periphery = valid & (distance > cutoff)

        centre_valid = int(centre.sum())
        periphery_valid = int(periphery.sum())

        ids, w = _grid_weights(valid)

        rows = []

        for target, values in [
            ("baseline_NEP", base_values),
            ("dNEP", delta_values),
        ]:
            cluster = _lisa_cluster_map(
                values,
                valid,
                ids,
                w,
                permutations=PERMUTATIONS,
            )

            for label, quadrant in [("HH", 1), ("LL", 3)]:
                centre_n = int((centre & (cluster == quadrant)).sum())
                periphery_n = int(
                    (periphery & (cluster == quadrant)).sum()
                )

                centre_rate = centre_n / centre_valid
                periphery_rate = periphery_n / periphery_valid

                rows.append({
                    "city": city,
                    "target": target,
                    "cluster": label,
                    "permutations": PERMUTATIONS,
                    "target_resolution_m": TARGET_RESOLUTION_M,
                    "actual_resolution_y_m": original_dy * factor_y,
                    "actual_resolution_x_m": original_dx * factor_x,
                    "factor_y": factor_y,
                    "factor_x": factor_x,
                    "valid_n": n_valid,
                    "centre_valid": centre_valid,
                    "periphery_valid": periphery_valid,
                    "centre_n": centre_n,
                    "periphery_n": periphery_n,
                    "centre_rate": centre_rate,
                    "periphery_rate": periphery_rate,
                    "rate_difference": (
                        periphery_rate - centre_rate
                    ),
                    "rate_ratio": (
                        periphery_rate / centre_rate
                        if centre_rate > 0 else np.nan
                    ),
                })

        return pd.DataFrame(rows)
    OUT_DIR = root_proj / '4_analysis' / f'extra_lisa_{TARGET_RESOLUTION_M}m_{PERMUTATIONS}perm'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [analyse_city_lisa(city) for city in tqdm(available_cities())]
    pd.concat(rows, ignore_index=True).to_csv(OUT_DIR / 'centre_periphery_rates.csv', index=False)


def run_socioeconomic():
    'v3 cells 100,101,104: WorldPop alignment, NTL extension and NEP co-trends; original exclusions retained.'
    country_to_iso3 = {
        "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Andorra": "AND",
        "Angola": "AGO", "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM",
        "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE",
        "Bahamas": "BHS", "Bahrain": "BHR", "Bangladesh": "BGD", "Barbados": "BRB",
        "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN", "Bhutan": "BTN",
        "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH", "Botswana": "BWA", "Brazil": "BRA",
        "Brunei": "BRN", "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI",
        "Cabo Verde": "CPV", "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN",
        "Central African Republic": "CAF", "Chad": "TCD", "Chile": "CHL", "China": "CHN",
        "Colombia": "COL", "Comoros": "COM", "Congo (Congo-Brazzaville)": "COG",
        "Costa Rica": "CRI", "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP", # "Czechia": "CZE",
        "Czech Republic": "CZE", # "Democratic Republic of the Congo": "COD",
        "DR Congo": "COD", "Denmark": "DNK", "Djibouti": "DJI", "Dominica": "DMA", "Dominican Republic": "DOM",
        "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV", "Equatorial Guinea": "GNQ",
        "Eritrea": "ERI", "Estonia": "EST", "Eswatini": "SWZ", "Ethiopia": "ETH",
        "Fiji": "FJI", "Finland": "FIN", "France": "FRA",
        "Gabon": "GAB", "Gambia": "GMB", "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA",
        "Greece": "GRC", "Grenada": "GRD", "Guatemala": "GTM", "Guinea": "GIN",
        "Guinea-Bissau": "GNB", "Guyana": "GUY",
        "Haiti": "HTI", "Honduras": "HND", "Hungary": "HUN", 'Hong Kong': 'HKG',
        "Iceland": "ISL", "India": "IND", "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ",
        "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA", "Ivory Coast": "CIV",
        "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR",
        "Kazakhstan": "KAZ", "Kenya": "KEN", "Kiribati": "KIR", "Kuwait": "KWT", "Kyrgyzstan": "KGZ",
        "Laos": "LAO", "Latvia": "LVA", "Lebanon": "LBN", "Lesotho": "LSO",
        "Liberia": "LBR", "Libya": "LBY", "Liechtenstein": "LIE", "Lithuania": "LTU", "Luxembourg": "LUX",
        "Madagascar": "MDG", "Malawi": "MWI", "Malaysia": "MYS", "Maldives": "MDV",
        "Mali": "MLI", "Malta": "MLT", "Marshall Islands": "MHL",
        "Mauritania": "MRT", "Mauritius": "MUS", "Mexico": "MEX", "Micronesia": "FSM",
        "Moldova": "MDA", "Monaco": "MCO", "Mongolia": "MNG", "Montenegro": "MNE",
        "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR",
        "Namibia": "NAM", "Nauru": "NRU", "Nepal": "NPL", "Netherlands": "NLD",
        "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER", "Nigeria": "NGA",
        "North Korea": "PRK", "North Macedonia": "MKD", "Norway": "NOR",
        "Oman": "OMN",
        "Pakistan": "PAK", "Palau": "PLW", "Panama": "PAN", "Papua New Guinea": "PNG",
        "Paraguay": "PRY", "Peru": "PER", "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT",
        "Qatar": "QAT",
        "Romania": "ROU", "Russia": "RUS", "Rwanda": "RWA",
        "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA", "Saint Vincent and the Grenadines": "VCT",
        "Samoa": "WSM", "San Marino": "SMR", "Sao Tome and Principe": "STP",
        "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB", "Seychelles": "SYC",
        "Sierra Leone": "SLE", "Singapore": "SGP", "Slovakia": "SVK", "Slovenia": "SVN",
        "Solomon Islands": "SLB", "Somalia": "SOM", "South Africa": "ZAF",
        "South Korea": "KOR", "South Sudan": "SSD", "Spain": "ESP", "Sri Lanka": "LKA",
        "Sudan": "SDN", "Suriname": "SUR", "Sweden": "SWE", "Switzerland": "CHE", "Syria": "SYR",
        "Taiwan": "TWN", "Tajikistan": "TJK", "Tanzania": "TZA", "Thailand": "THA",
        "Timor-Leste": "TLS", "Togo": "TGO", "Tonga": "TON", "Trinidad and Tobago": "TTO",
        "Tunisia": "TUN", "Turkey": "TUR", "Turkmenistan": "TKM", "Tuvalu": "TUV",
        "Uganda": "UGA", "Ukraine": "UKR", # "United Arab Emirates": "ARE",
        "UAE": "ARE", "United Kingdom": "GBR", # "United States": "USA",
        "USA": "USA", "Uruguay": "URY", "Uzbekistan": "UZB",
        "Vanuatu": "VUT", "Vatican City": "VAT", "Venezuela": "VEN",
        "Vietnam": "VNM", "Yemen": "YEM",
        "Zambia": "ZMB", "Zimbabwe": "ZWE"
    }
    df_path_pop = []
    for p in root_proj.joinpath('1_Worldpop').glob('*.tif'):
        country_code_, _, year = p.stem.split('_')
        dt = pd.to_datetime(year, format = '%Y')
        df_path_pop.append([country_code_, dt, p])
    df_path_pop = pd.DataFrame(df_path_pop, columns = ['country_code', 'time', 'path']).set_index(['country_code', 'time']).sort_index()
    from sklearn.preprocessing import StandardScaler

    savefolder = root_proj.joinpath('4_analysis/UrbanEcoCorr')
    savefolder.mkdir(parents=True, exist_ok=True)

    suffix = '_EVI_DynamicWorld_UFLUXv1'
    city_names = sorted([p.stem for p in root_proj.joinpath(f'3_output{suffix}').glob('*') if not p.is_file()])

    for city_name in tqdm(meta_city.index, desc = "city", position = 0):
        if city_name == 'Dakar': continue
        savefile = savefolder.joinpath(f'{city_name}_percap_corr.csv')
        if savefile.exists(): continue
        country_code = country_to_iso3[meta_city.loc[city_name, 'country']]

        period1 = ['2016', '2020']; period2 = ['2021', '2025']

        d_dataset = load_city_uflux(city_name, suffix, period1 = period1, period2 = period2)
        ncu = d_dataset['uflux']
        ncu_avg = d_dataset['uflux_avg']
        ncu0 = d_dataset['uflux_period0']
        ncu0_avg = d_dataset['uflux_period0_avg']
        ncu1 = d_dataset['uflux_period1']
        ncu1_avg = d_dataset['uflux_period1_avg']
        ncd = d_dataset['uflux_periods_diff']
        ncd_lucc = d_dataset['uflux_periods_diff_lucc']

        shp = gpd.read_file(root_proj.joinpath(f"2_geojsons/{city_name}.geojson"), driver='GeoJSON')
        minx, miny, maxx, maxy = shp.total_bounds
        minx -= 0.1; miny -= 0.1; maxx += 0.1; maxy += 0.1


        nc_pop = []
        for yr in tqdm(range(2016, 2026), desc = "Year", position = 1, leave = False):
            p = df_path_pop.loc[(country_code, f'{yr}-01-01'), 'path']
            nc_pop_t = geoface.load_tif(p, band_names = ['population'])['population']
            lat = nc_pop_t.latitude.values
            lon = nc_pop_t.longitude.values
            if lat[0] > lat[-1]:
                lat_slice = slice(maxy, miny)   # descending latitude
            else:
                lat_slice = slice(miny, maxy)

            if lon[0] > lon[-1]:
                lon_slice = slice(maxx, minx)   # descending longitude (rare but possible)
            else:
                lon_slice = slice(minx, maxx)

            nc_pop_t = nc_pop_t.sel(
                latitude=lat_slice,
                longitude=lon_slice
            )
            nc_pop_t = geoface.clip(nc_pop_t, shp)
            nc_pop_t = nc_pop_t.where(nc_pop_t != -99999.)
            nc_pop_t = nc_pop_t.expand_dims(time = [pd.to_datetime(f'{yr}-01-01')])
            nc_pop.append(nc_pop_t)
        nc_pop = xr.merge(nc_pop).interp(latitude = ncu.latitude, longitude = ncu.longitude).rio.write_crs("epsg:4326", inplace = False).rio.clip(shp.geometry.values, shp.crs).drop_vars('spatial_ref')


        nc_pop['population'] = nc_pop['population'].where(
            nc_pop['population'] != -99999,
            np.nan
        )


        if root_proj.joinpath(f'1_griddata/{city_name}').exists():
            city_folder = root_proj.joinpath(f'1_griddata/{city_name}')
        else:
            city_folder = root_proj.joinpath(f'1_griddata_202605/{city_name}')

        df_path_aqua_temp = []
        for p in city_folder.joinpath(f'{city_name}_Aqua_temperature_1km').glob('*.tif'):
            df_path_aqua_temp.append([pd.to_datetime(p.stem), p])
        df_path_aqua_temp = pd.DataFrame(df_path_aqua_temp, columns = ['time', 'path']).set_index(['time']).sort_index()

        df_path_terra_temp = []
        for p in city_folder.joinpath(f'{city_name}_Terra_temperature_1km').glob('*.tif'):
            df_path_terra_temp.append([pd.to_datetime(p.stem), p])
        df_path_terra_temp = pd.DataFrame(df_path_terra_temp, columns = ['time', 'path']).set_index(['time']).sort_index()

        df_path_nightlight = []
        for p in city_folder.joinpath(f'{city_name}_VIIRS_Nightlight_500m').glob('*.tif'):
            df_path_nightlight.append([pd.to_datetime(p.stem), p])
        df_path_nightlight = pd.DataFrame(df_path_nightlight, columns = ['time', 'path']).set_index(['time']).sort_index()


        nc_aqua = xr.merge([geoface.load_tif(df_path_aqua_temp.loc[dt, 'path'], band_names=['Ta']).expand_dims(time = [dt]) - 273.15 for dt in df_path_aqua_temp.index])
        nc_terra = xr.merge([geoface.load_tif(df_path_terra_temp.loc[dt, 'path'], band_names=['Ta']).expand_dims(time = [dt]) - 273.15 for dt in df_path_terra_temp.index])
        nc_temp = (nc_aqua + nc_terra) / 2; del(nc_aqua, nc_terra)
        nc_nightlight = xr.merge([geoface.load_tif(df_path_nightlight.loc[dt, 'path'], band_names=['nightlight']).expand_dims(time = [dt]) for dt in df_path_nightlight.index])


        years = nc_nightlight.time.dt.year.values

        da = nc_nightlight["nightlight"]
        da = da[da.time.dt.year < 2025]

        fit = da.polyfit(dim="time", deg=1)

        time_2025 = np.datetime64("2025-01-01")

        pred_2025 = xr.polyval(
            coord=xr.DataArray(time_2025, dims=[]),
            coeffs=fit.polyfit_coefficients
        )

        pred_2025 = pred_2025.expand_dims(
            time=[time_2025]
        ).astype("float32")

        nc_nightlight_extended = xr.concat(
            [da, pred_2025],
            dim="time"
        )

        nc_nightlight_extended = nc_nightlight_extended.to_dataset(name="nightlight")


        nc_nightlight_extended = nc_nightlight_extended.interp(latitude = ncu.latitude, longitude = ncu.longitude).rio.write_crs("epsg:4326", inplace = False).rio.clip(shp.geometry.values, shp.crs).drop_vars('spatial_ref')
        nc_temp = nc_temp.interp(latitude = ncu.latitude, longitude = ncu.longitude).rio.write_crs("epsg:4326", inplace = False).rio.clip(shp.geometry.values, shp.crs).drop_vars('spatial_ref')



        ncc = xr.merge([ncu, nc_pop, nc_nightlight_extended, nc_temp])


        df = pd.concat([
            (ncc['EVI'] / ncc['population']).median(dim=['latitude','longitude']).rename('EVI (per cap)').to_dataframe(),
            (ncc['NEP'] / ncc['population']).median(dim=['latitude','longitude']).rename('NEP (per cap)').to_dataframe(),
            (ncc['LUE'] / ncc['population']).median(dim=['latitude','longitude']).rename('LUE (per cap)').to_dataframe(),
            xr.corr(ncc['nightlight'], ncc['population'], dim=['latitude','longitude']).rename('NTL_Pop corr').to_dataframe(),
            xr.corr(ncc['Ta'], ncc['nightlight'], dim=['latitude','longitude']).rename('TA_NTL corr').to_dataframe(),
        ], axis=1)

        nc = xr.merge([
            xr.corr(ncc['nightlight'].diff('time'), ncc['NEP'].diff('time'), dim='time').rename('NTL_NEP diff corr'),
            xr.corr(ncc['nightlight'].diff('time'), ncc['population'].diff('time'), dim='time').rename('NTL_Pop diff corr'),
        ])

        df.to_csv(savefile)
        nc.to_netcdf(savefolder.joinpath(f'{city_name}_diff_corr.nc'))


def run_tables():
    """Cross-city NEP, land-cover fractions and LUCC summaries (v3 cells 42,50,51,54)."""
    # Source: v3 cell 42.
    df_path = []
    for p in root_proj.joinpath('4_analysis').rglob('UFLUX_multiyear_average.csv'):
        df_path.append([p.parent.stem, p])
    df_path = pd.DataFrame(df_path, columns = ['CITY', 'PATH']).set_index('CITY')

    # ------------------------------------------------------------------------------

    dfp = []
    for city_name in df_path.index:
        p_avg = df_path.loc[city_name, 'PATH']
        df_avg = pd.read_csv(p_avg, index_col = 0).loc['Avg(2016-2025)', :].rename(city_name).to_frame().T

        df_avg_lc = pd.read_csv(p_avg.parent.joinpath('UFLUX_multiyear_average_LC.csv'), index_col = 0)
        df_avg_lc.index = df_avg_lc.index.map(d_dw_lc)
        # df_avg['rGreen'] = df_avg_lc.loc[df_avg_lc.index.intersection([1, 2, 4, 5]), '%pixel'].sum()
        df_avg['rGreen'] = df_avg_lc.loc[df_avg_lc.index.intersection(['trees', 'grass', 'crops', 'shrub_and_scrub']), '%pixel'].sum()

        dfp.append(df_avg)

    dfp = pd.concat(dfp)
    dfp['Rw'] = dfp['Rw'].clip(0, 1.5)

    dfp.index = dfp.index.to_series()
    dfp = dfp.sort_index()

    dfp['Continent'] = dfp.index.map(city_continents)
    dfp = dfp[dfp.index != 'Dakar']
    # Source: v3 cell 50.
    df_path = []
    for p in root_proj.joinpath('4_analysis').rglob('UFLUX_multiyear_diff_LUCC_sum.csv'):
        df_path.append([p.parent.stem, p])
    df_path = pd.DataFrame(df_path, columns = ['CITY', 'PATH']).set_index('CITY')

    # ------------------------------------------------------------------------------

    dfd_avg = []; dfd_lucc_avg = []; dfd_nonlucc_avg = []; dfd_dueTolucc = []
    for city_name in tqdm(df_path.index):
        if city_name in ['Accra', 'Dakar']: continue
        p_lucc_sum = df_path.loc[city_name, 'PATH']
        p_lucc = p_lucc_sum.parent.joinpath('UFLUX_multiyear_diff_LUCC.csv')

        df_lucc_sum = pd.read_csv(p_lucc_sum, index_col = 0).drop('LUCC', axis = 1)
        df_lucc = pd.read_csv(p_lucc, index_col = 0)
        df_lucc_sum['rLUCC'] = np.nan

        df_lucc_sum.loc['No change sum', 'rLUCC'] = df_lucc.loc[0, '%pixel']
        df_lucc_sum.loc['LUCC sum', 'rLUCC'] = 1 - df_lucc.loc[0, '%pixel']
        df_lucc_sum.loc['LUCC only sum', 'rLUCC'] = 1 - df_lucc.loc[0, '%pixel']
        df_lucc_sum.loc['Avg sum', 'rLUCC'] = 1

        dfd_avg.append(df_lucc_sum.loc['Avg sum', :].rename(city_name).to_frame().T)
        dfd_lucc_avg.append(df_lucc_sum.loc['LUCC sum', :].rename(city_name).to_frame().T)
        dfd_dueTolucc.append(df_lucc_sum.loc['LUCC only sum', :].rename(city_name).to_frame().T)
        dfd_nonlucc_avg.append(df_lucc_sum.loc['No change sum', :].rename(city_name).to_frame().T)

    dfd_avg = pd.concat(dfd_avg)
    dfd_lucc_avg = pd.concat(dfd_lucc_avg)
    dfd_dueTolucc = pd.concat(dfd_dueTolucc)
    dfd_nonlucc_avg = pd.concat(dfd_nonlucc_avg)

    dfd = pd.concat([
        dfd_avg['NEP'].rename('dNEP'),
        dfd_lucc_avg['NEP'].rename('dNEP_LUCC'),
        dfd_dueTolucc['NEP'].rename('dNEP_dueToLUCC'),
        dfd_nonlucc_avg['NEP'].rename('dNEP_nonLUCC'),
        dfd_lucc_avg['rLUCC']
    ], axis = 1)

    if not np.allclose((dfd['dNEP_LUCC'] * dfd['rLUCC'] + dfd['dNEP_nonLUCC'] * (1 - dfd['rLUCC']) - dfd['dNEP']).round(4), 0):
        raise ValueError('The sum of LUCC + nonLUCC changes should be equal to all changes!')
    # Source: v3 cell 51.
    df_path = []
    for p in root_proj.joinpath('4_analysis').rglob('UFLUX_multiyear_diff_LUCC_only.csv'):
        df_path.append([p.parent.stem, p])
    df_path = pd.DataFrame(df_path, columns = ['CITY', 'PATH']).set_index('CITY')

    # ------------------------------------------------------------------------------

    # 0: Unchanged;
    # 1: Urbanisation;
    # 2: Agricultural expansion;
    # 3: Agricultural abandonment;
    # 4: Flooding / inundation;
    # 5: Greening (incl. afforestation);
    # 6: Degradation / clearing;
    # 7: Other change

    dfd_dueTolucc_cat = []
    for city_name in tqdm(df_path.index):
        p_dueTolucc = df_path.loc[city_name, 'PATH']
        df_dueTolucc = pd.read_csv(p_dueTolucc, index_col = 0) # LUCC == 0 only include water <-> flooded
        df_dueTolucc = df_dueTolucc[df_dueTolucc.index != 0]

        dfd_dueTolucc_cat.append(df_dueTolucc['NEP'].rename(city_name).to_frame().T)

    dfd_dueTolucc_cat = pd.concat(dfd_dueTolucc_cat, axis = 0)

    dfd_dueTolucc_cat = dfd_dueTolucc_cat.rename(columns = {
        0: "Unchanged",
        1: "Urbanisation",
        2: "Agri.exp.",
        3: "Agri.aband.",
        4: "Inundation",
        5: "Greening",
        6: "Degradation",
        7: "Others"
    })
    # Source: v3 cell 54.
    df_path = []
    for p in root_proj.joinpath('4_analysis').rglob('UFLUX_multiyear_average_LC.csv'):
        df_path.append([p.parent.stem, p])
    df_path = pd.DataFrame(df_path, columns = ['CITY', 'PATH']).set_index('CITY')

    # ------------------------------------------------------------------------------

    df_lc_data = []; df_lc_rpixel = []
    for city_name in tqdm(df_path.index):
        if city_name in ['Accra', 'Dakar']: continue
        p_lc = df_path.loc[city_name, 'PATH']
        df_lc = pd.read_csv(p_lc, index_col = 0)
        # df_lc = df_lc[df_lc.index != 0]
        # df_lc['%pixel'] = df_lc['%pixel'] / df_lc['%pixel'].sum()

        df_lc_data.append(df_lc['NEP'].rename(city_name).to_frame().T)
        df_lc_rpixel.append(df_lc['%pixel'].rename(city_name).to_frame().T)

    df_lc_data = pd.concat(df_lc_data, axis = 0).rename(columns = {
        0: "water",
        1: "trees",
        2: "grass",
        3: "flooded",
        4: "crops",
        5: "shrub",
        6: "built",
        7: "bare",
        8: "snowice"
    })
    df_lc_rpixel = pd.concat(df_lc_rpixel, axis = 0).rename(columns = {
        0: "water",
        1: "trees",
        2: "grass",
        3: "flooded",
        4: "crops",
        5: "shrub",
        6: "built",
        7: "bare",
        8: "snowice"
    })

    df_lc_data.index = df_lc_data.index.to_series()
    df_lc_data = df_lc_data.sort_index()
    df_lc_data = df_lc_data.abs()
    df_lc_data['Continent'] = df_lc_data.index.map(city_continents)

    df_lc_rpixel.index = df_lc_rpixel.index.to_series()
    df_lc_rpixel = df_lc_rpixel.sort_index()
    df_lc_rpixel['Continent'] = df_lc_rpixel.index.map(city_continents)
    export_dir = root_proj / '5_supp' / 'cross_city_tables'
    export_dir.mkdir(parents=True, exist_ok=True)
    for name, value in list(locals().items()):
        if isinstance(value, pd.DataFrame) and name != 'df_path':
            value.to_csv(export_dir / (name + '.csv'))


def available_cities():
    folder = root_proj / '4_analysis/spatial_EVI_DynamicWorld_UFLUXv1'
    cities = sorted(p.name.removesuffix('_uflux.nc') for p in folder.glob('*_uflux.nc'))
    if not cities:
        raise FileNotFoundError(f'No annual analysis grids found in {folder}')
    return cities


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, help='Local urban_sust data directory')
    parser.add_argument('--stage', choices=['tables','summary','drivers','tree','robustness','ablation','spatial','socioeconomic'], help='Run one analysis workflow')
    args = parser.parse_args()
    if args.data_root is None or args.stage is None:
        parser.error('--data-root and --stage are required (use --help for input layout)')
    global root_proj
    root_proj = args.data_root.expanduser().resolve()
    if not root_proj.is_dir():
        parser.error(f'Data directory does not exist: {root_proj}')
    _load_dependencies(args.stage)
    _init_metadata()
    np.random.seed(42)
    if args.stage in {'summary','drivers','ablation'}:
        warnings.warn('Extracted source estimators retain the evaluation/formula limitations documented in this file.', stacklevel=1)
    globals()['run_' + args.stage]()


if __name__ == '__main__':
    main()

