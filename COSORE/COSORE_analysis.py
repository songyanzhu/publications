"""Analysis functions extracted from COSORE.ipynb and Xiao_site_representativeness_IAV.ipynb.

Inputs are already processed model outputs and observation tables. This file does
not download data, train UFLUX prediction models, or draw manuscript figures.
Flux units are explicit in each function; inspect source products before reuse.

Example (in the project directory)::

    python COSORE_analysis.py --corrected-dir 3_output/UFLUXv1_corrected

Other functions can be imported for analyses that need additional inputs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


EARTH_RADIUS_M = 6_371_000.0


def grid_cell_area(latitude: xr.DataArray, longitude: xr.DataArray) -> xr.DataArray:
    """Area (m2) of each regular latitude/longitude grid cell."""
    dlat = abs(float(latitude[1] - latitude[0]))
    dlon = abs(float(longitude[1] - longitude[0]))
    lat_area = EARTH_RADIUS_M**2 * np.deg2rad(dlon) * (
        np.sin(np.deg2rad(latitude + dlat / 2))
        - np.sin(np.deg2rad(latitude - dlat / 2))
    )
    return lat_area.broadcast_like(
        xr.DataArray(
            np.empty((latitude.size, longitude.size)),
            dims=("latitude", "longitude"),
            coords={"latitude": latitude, "longitude": longitude},
        )
    )


def validation_metrics(observed, predicted) -> dict[str, float]:
    """Paired record metrics; R2 uses the coefficient-of-determination definition."""
    x, y = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2:
        raise ValueError("At least two finite paired observations are required")
    err = y - x
    sst = np.sum((x - x.mean()) ** 2)
    return {
        "N": len(x), "bias": float(err.mean()),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(1 - np.sum(err**2) / sst) if sst > 0 else np.nan,
    }


def ratio_quality_control(frame: pd.DataFrame, rs="Rs", reco="RECO"):
    """Classify paired chamber Rs and predicted RECO; retain 0 <= Rs <= RECO."""
    paired = frame.loc[np.isfinite(frame[rs]) & np.isfinite(frame[reco])].copy()
    category = np.select(
        [paired[reco] <= 0, paired[rs] < 0, paired[rs] > paired[reco]],
        ["RECO <= 0", "Rs < 0", "Rs > RECO"],
        default="valid",
    )
    counts = pd.Series(category).value_counts().reindex(
        ["RECO <= 0", "Rs < 0", "Rs > RECO", "valid"], fill_value=0
    )
    summary = pd.DataFrame({"records": counts, "percent_of_paired": counts / len(paired) * 100})
    valid = paired.loc[category == "valid"].copy()
    valid["Rs2RECO"] = valid[rs] / valid[reco]
    return valid, summary


def ratio_validation(frame: pd.DataFrame, obs="Chamber_Rs2RECO", pred="UFLUX_Rs2RECO"):
    """Distribution test and paired validation on the *same* finite records."""
    matched = frame[[obs, pred]].dropna()
    metrics = validation_metrics(matched[obs], matched[pred])
    ks = stats.ks_2samp(matched[obs], matched[pred])
    metrics.update(
        KS_D=float(ks.statistic), KS_p=float(ks.pvalue),
        observed_mean=float(matched[obs].mean()),
        modeled_mean=float(matched[pred].mean()),
        observed_sd=float(matched[obs].std()),
        modeled_sd=float(matched[pred].std()),
    )
    # The notebook's validation scatter annotated Pearson r squared, which is
    # distinct from the coefficient-of-determination R2 above.
    metrics["Pearson_r_squared"] = float(stats.pearsonr(matched[obs], matched[pred]).statistic**2)
    return metrics


def srdb_grid_year(frame: pd.DataFrame, resolution=0.5) -> pd.DataFrame:
    """Average SRDBv5 annual observations and UFLUX predictions by grid-year.

    Columns follow source cell 79: Study_midyear, latitude, longitude,
    Site_ID, Rs_annual and Rs. Rs_annual and Rs are g C m-2 yr-1.
    """
    df = frame.copy()
    df["year"] = pd.to_datetime(df.Study_midyear).dt.year
    for axis in ("latitude", "longitude"):
        df[axis + "_bin"] = np.floor(df[axis] / resolution) * resolution
    result = df.groupby(["year", "latitude_bin", "longitude_bin"], as_index=False).agg(
        Rs_annual=("Rs_annual", "mean"), Rs=("Rs", "mean"),
        n_records=("Site_ID", "count"),
    )
    return result.loc[result.Rs != 0]


def binned_driver_response(frame: pd.DataFrame, driver: str, bins=12) -> pd.DataFrame:
    """Observed/modeled ratio response over the central 1st--99th percentile.

    Matches the quantile-bin setup of notebook cell 140 without drawing a figure.
    Precipitation in source data is m d-1; multiply the bin centres by 1000
    when labeling a result in mm d-1.
    """
    observed, modeled = "Chamber_Rs2RECO", "UFLUX_Rs2RECO"
    df = frame[[driver, observed, modeled]].dropna().copy()
    low, high = df[driver].quantile([0.01, 0.99])
    df = df.loc[df[driver].between(low, high)]
    df["bin"] = pd.qcut(df[driver], q=bins, duplicates="drop")
    result = df.groupby("bin", observed=True).agg(
        observed=(observed, "mean"), modeled=(modeled, "mean"), n=(observed, "size")
    )
    result["driver_mid"] = [interval.mid for interval in result.index]
    return result.reset_index(drop=True)


def annual_flux_pgC(rate: xr.DataArray, mode="annual_mean") -> xr.DataArray:
    """Integrate g C m-2 d-1 to Pg C yr-1 (calendar year or monthly rates)."""
    area = grid_cell_area(rate.latitude, rate.longitude)
    if mode == "annual_mean":
        flux = rate * rate.time.dt.days_in_year
    elif mode == "monthly_mean":
        flux = (rate * rate.time.dt.days_in_month).groupby("time.year").sum("time")
    else:
        raise ValueError("mode must be annual_mean or monthly_mean")
    result = (flux * area).sum(("latitude", "longitude")) / 1e15
    result.attrs["units"] = "Pg C yr-1"
    return result


def corrected_annual_series(directory: str | Path):
    """Fixed-mask 2000-2024 totals from yearly corrected UFLUX files.

    Source convention in Xiao's notebook: Rs2 = annual cumulative g C m-2
    yr-1; GPP and RECO = mean daily g C m-2 d-1, multiplied by 365.
    """
    paths = {}
    for path in Path(directory).glob("*.nc"):
        match = re.fullmatch(r"(20\d\d)-UFLUXv1-Rs\.nc", path.name)
        if match and 2000 <= int(match[1]) <= 2024:
            paths[int(match[1])] = path
    if len(paths) != 25:
        raise ValueError(f"Expected 25 annual files (2000-2024); found {sorted(paths)}")
    cubes = {name: [] for name in ("Rs2", "GPP", "RECO")}
    lat = lon = None
    for year, path in sorted(paths.items()):
        with xr.open_dataset(path) as ds:
            if lat is None:
                lat, lon = ds.latitude.values.copy(), ds.longitude.values.copy()
            elif not (np.array_equal(lat, ds.latitude) and np.array_equal(lon, ds.longitude)):
                raise ValueError(f"Grid mismatch in {path}")
            for name in cubes:
                field = ds[name]
                if "time" in field.dims:
                    if field.sizes["time"] != 1:
                        raise ValueError(f"Expected one annual {name} field in {path}")
                    field = field.isel(time=0, drop=True)
                cubes[name].append(field.transpose("latitude", "longitude").values.copy())
    for name in cubes:
        cubes[name] = np.stack(cubes[name])
    mask = np.logical_and.reduce([
        np.isfinite(a).all(axis=0) & (a >= 0).all(axis=0) for a in cubes.values()
    ])
    area = grid_cell_area(xr.DataArray(lat, dims="latitude"),
                          xr.DataArray(lon, dims="longitude")).values
    years = np.array(sorted(paths))
    series = pd.DataFrame(index=pd.Index(years, name="year"))
    for name, a in cubes.items():
        factor = 1 if name == "Rs2" else 365  # Source convention, not leap-day scaling.
        series[name] = np.sum(np.where(mask[None, :, :], a, 0) * area[None, :, :],
                              axis=(1, 2)) * factor / 1e15
    trend = stats.linregress(years, series.Rs2)
    return series, mask, {"slope_PgC_yr2": trend.slope, "p": trend.pvalue,
                          "r2": trend.rvalue**2, "valid_cells": int(mask.sum())}


def map_fusion(
    cosore_daily: xr.DataArray,
    srdb_daily: xr.DataArray,
    validation: pd.DataFrame,
    test_index: pd.Index,
):
    """Notebook's inverse-MSE weighted fusion of *multi-year mean daily* maps.

    Validation columns Rs_meas, Rs_pred and Rs_pred2 are annual g C m-2 yr-1.
    test_index must be the held-out SRDBv5 residual-learning indices; the
    function does not infer a split or assume independent model errors.
    """
    heldout = validation.loc[test_index, ["Rs_meas", "Rs_pred", "Rs_pred2"]].dropna()
    obs_error = (heldout.Rs_pred - heldout.Rs_meas) / 365.25
    prior_error = (heldout.Rs_pred2 - heldout.Rs_meas) / 365.25
    sigma2, tau2 = float(np.mean(obs_error**2)), float(np.mean(prior_error**2))
    if min(sigma2, tau2) <= 0:
        raise ValueError("Both held-out MSEs must be positive")
    obs, prior = xr.align(cosore_daily, srdb_daily, join="exact")
    post = (prior / tau2 + obs / sigma2) / (1 / tau2 + 1 / sigma2)
    return post.rename("Rs_MAP"), {
        "sigma2": sigma2, "tau2": tau2,
        "COSORE_weight": (1 / sigma2) / (1 / sigma2 + 1 / tau2),
        "SRDBv5_weight": (1 / tau2) / (1 / sigma2 + 1 / tau2),
        "input_error_correlation": float(np.corrcoef(obs_error, prior_error)[0, 1]),
    }


def bootstrap_mean_totals(series: pd.DataFrame, n_boot=10_000, seed=42):
    """Paired year bootstrap of product means, using one fixed spatial mask.

    Input columns are annual global totals in Pg C yr-1 (e.g. COSORE, SRDBv5,
    MAP). All products use the same sampled years in each bootstrap replicate.
    """
    if series.isna().any().any() or len(series) < 2:
        raise ValueError("Supply at least two complete common years")
    array = series.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_boot, len(array)))
    boot = array[indices].mean(axis=1)
    return pd.DataFrame({"mean": array.mean(axis=0),
                         "CI95_low": np.percentile(boot, 2.5, axis=0),
                         "CI95_high": np.percentile(boot, 97.5, axis=0)},
                        index=series.columns)


def huang_comparison(uflux_annual: xr.DataArray, huang_annual: xr.DataArray):
    """Compare annual g C m-2 yr-1 products on shared years and fixed grid.

    Supply the *pre-coarsened* Huang 0.5 degree file; do not open the 67 GB
    source here. Coordinate names must be time/latitude/longitude in both.
    """
    u, h = xr.align(uflux_annual, huang_annual, join="inner")
    if u.sizes.get("time", 0) < 2:
        raise ValueError("Products have fewer than two common years")
    valid = (np.isfinite(u) & np.isfinite(h) & (u >= 0) & (h >= 0)).all("time")
    um, hm = u.where(valid).mean("time"), h.where(valid).mean("time")
    area = grid_cell_area(u.latitude, u.longitude)
    yearly = pd.DataFrame({
        "UFLUX_PgC_yr": ((u.where(valid) * area).sum(("latitude", "longitude")) / 1e15).values,
        "Huang_PgC_yr": ((h.where(valid) * area).sum(("latitude", "longitude")) / 1e15).values,
    }, index=pd.Index(u.time.dt.year.values, name="year"))
    a, b = um.values[valid.values], hm.values[valid.values]
    return yearly, {"spatial_r": float(stats.pearsonr(a, b).statistic),
                    "valid_cells": len(a)}, um, hm


def site_vegetation_shares(
    site_locations: pd.DataFrame, land_cover: xr.DataArray,
    valid_land: xr.DataArray,
) -> pd.DataFrame:
    """Site versus area shares for matched unique sites (Xiao notebook cell 1).

    site_locations has columns stream and IGBP after climate/land-cover matching;
    land_cover is the 2010 IGBP grid, with class 14 already mapped to 12.
    """
    area = grid_cell_area(land_cover.latitude, land_cover.longitude)
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    land = pd.DataFrame({"code": land_cover.where(valid_land).values.ravel(),
                         "area": area.values.ravel()}).dropna()
    land = land.loc[land.code.isin(classes)]
    shares = land.groupby("code").area.sum().reindex(classes, fill_value=0)
    shares = shares / shares.sum() * 100
    sites = pd.crosstab(site_locations.IGBP, site_locations.stream,
                        normalize="columns") * 100
    return pd.DataFrame({"Land area": shares}).join(sites, how="left").fillna(0)


def causal_vpd(frame: pd.DataFrame):
    """Notebook's CausalForestDML VPD sensitivity analysis; no plotting.

    Requires optional econml and lightgbm packages. Returns per-record
    conditional effects and biome summaries (record-weighted mean and SE).
    This observational analysis does not establish an experimental effect.
    """
    from econml.dml import CausalForestDML
    from lightgbm import LGBMRegressor

    confounders = ["soil_temperature_level_1", "surface_solar_radiation_downwards_sum",
                   "total_precipitation_sum", "GPP", "EVI_500m_MODIS", "DEM", "MONTH", "IGBP"]
    columns = ["VPD", "Chamber_Rs2RECO", *confounders]
    data = frame.loc[(frame.temperature_2m <= 30) &
                     (frame.soil_temperature_level_1 <= 30), columns].dropna().copy()
    data = data.loc[~data.IGBP.isin(["DWL", "CSH"])]  # As in source cell 156.
    x = pd.get_dummies(data[confounders], columns=["IGBP"], drop_first=True)
    forest = CausalForestDML(
        model_y=LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1),
        model_t=LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1),
        discrete_treatment=False, n_estimators=500, random_state=42,
    )
    forest.fit(data.Chamber_Rs2RECO.to_numpy(), data.VPD.to_numpy(), X=x)
    data["CATE_VPD"] = forest.effect(x).ravel()
    group = data.groupby("IGBP").CATE_VPD.agg(["mean", "std", "count"])
    group["se"] = group["std"] / np.sqrt(group["count"])
    return data, group, forest


def main():
    parser = argparse.ArgumentParser(description="Summarize corrected UFLUX annual fluxes")
    parser.add_argument("--corrected-dir", type=Path, required=True,
                        help="Folder containing 2000-UFLUXv1-Rs.nc ... 2024-UFLUXv1-Rs.nc")
    args = parser.parse_args()
    series, _, trend = corrected_annual_series(args.corrected_dir)
    print(series.round(3).to_string())
    print("\nTrend and common coverage:", trend)


if __name__ == "__main__":
    main()
