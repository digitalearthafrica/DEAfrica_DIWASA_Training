from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from deafrica_tools.datahandling import load_ard


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_band(
    image: xr.Dataset | xr.DataArray,
    ax: plt.Axes,
    title: str,
    band: str = "green",
    scale_max: float = 3000,
    cmap: str = "gray",
) -> None:
    """Plot one image band with simple 0–255 scaling."""
    data = image[band] if isinstance(image, xr.Dataset) else image

    if "y" in data.coords:
        data = data.sortby("y", ascending=False)

    scaled = data.clip(min=0, max=scale_max) / scale_max * 255

    ax.imshow(scaled, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


def plot_mndwi_slices(
    data: xr.DataArray,
    title: str,
    output_dir: str | Path = RESULTS_DIR,
    col_wrap: int = 4,
    show: bool = True,
) -> Path:
    """Plot MNDWI time slices and save the figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot = data.plot(
        col="time",
        col_wrap=col_wrap,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
    )
    plot.fig.suptitle(title, fontsize=16, y=1.02)

    output_path = output_dir / f"{_safe_filename(title)}.png"
    plot.fig.savefig(output_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    plt.close(plot.fig)
    return output_path


def calculate_water_area_otsu(
    mndwi: xr.Dataset | xr.DataArray | np.ndarray,
    spatial_resolution: float = 30,
    variable: str = "MNDWI",
) -> pd.Series:
    """
    Calculate water area for each time slice using Otsu thresholding.

    NaN and infinite values are excluded before normalization.
    """
    if isinstance(mndwi, xr.Dataset):
        data_array = mndwi[variable]
    elif isinstance(mndwi, xr.DataArray):
        data_array = mndwi
    else:
        values = np.asarray(mndwi)
        if values.ndim == 2:
            values = values[np.newaxis, ...]
        data_array = xr.DataArray(
            values,
            dims=("time", "y", "x"),
            coords={"time": np.arange(values.shape[0])},
        )

    if "time" not in data_array.dims:
        data_array = data_array.expand_dims(time=[0])

    pixel_area_km2 = spatial_resolution**2 / 1e6
    areas: list[float] = []

    for time_slice in data_array:
        values = np.asarray(time_slice.values, dtype=np.float32)
        valid = np.isfinite(values)

        if not valid.any():
            areas.append(np.nan)
            continue

        valid_values = values[valid]

        if np.nanmin(valid_values) == np.nanmax(valid_values):
            areas.append(0.0)
            continue

        normalized = cv2.normalize(
            valid_values,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        threshold, _ = cv2.threshold(
            normalized,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        water_pixel_count = np.count_nonzero(normalized > threshold)
        areas.append(water_pixel_count * pixel_area_km2)

    return pd.Series(
        areas,
        index=pd.Index(data_array["time"].values, name="time"),
        name="Water Area",
    )


def calculate_rll(
    lake_surface_area: float | np.ndarray | pd.Series,
    a: float,
    b: float,
    c: float,
    root: str = "positive",
):
    """
    Solve a*RLL² + b*RLL + c = lake_surface_area.

    Parameters
    ----------
    root:
        ``"positive"`` uses +sqrt(discriminant);
        ``"negative"`` uses -sqrt(discriminant).
    """
    if a == 0:
        raise ValueError("Coefficient 'a' must not be zero.")

    lsa = np.asarray(lake_surface_area, dtype=float)
    discriminant = b**2 - 4 * a * (c - lsa)

    sqrt_discriminant = np.sqrt(np.where(discriminant >= 0, discriminant, np.nan))
    sign = 1 if root == "positive" else -1 if root == "negative" else None

    if sign is None:
        raise ValueError("root must be either 'positive' or 'negative'.")

    result = (-b + sign * sqrt_discriminant) / (2 * a)

    if np.isscalar(lake_surface_area):
        return float(result)
    if isinstance(lake_surface_area, pd.Series):
        return pd.Series(result, index=lake_surface_area.index)
    return result


def calculate_rll_from_area(
    file_path: str | Path,
    a: float,
    b: float,
    c: float,
    datum: float,
    output_path: str | Path = RESULTS_DIR / "Height_Cal_awi.csv",
    area_column: str = "Water Area",
    date_column: str = "time",
) -> pd.DataFrame:
    """Calculate lake height from lake-area observations and save the result."""
    data = pd.read_csv(file_path)

    required_columns = {area_column, date_column}
    missing = required_columns - set(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    result = pd.DataFrame(
        {
            "Date": pd.to_datetime(data[date_column], errors="coerce"),
            "Height_cal": calculate_rll(data[area_column], a, b, c) + datum,
        }
    ).dropna()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return result


def merge_and_process_height_data(
    file_path_alt: str | Path,
    file_path_cal: str | Path,
    output_path: str | Path = RESULTS_DIR / "Lake_Height_Ext_awi.csv",
    missing_value: float = 9999.99,
) -> pd.DataFrame:
    """
    Merge observed and calculated lake heights.

    Observed heights are preferred unless they equal the missing-value code.
    """
    observed = pd.read_csv(file_path_alt, parse_dates=["Date"]).rename(
        columns={"Height": "Height_observed"}
    )
    calculated = pd.read_csv(file_path_cal, parse_dates=["Date"]).rename(
        columns={"Height_cal": "Height_calculated"}
    )

    merged = observed.merge(calculated, on="Date", how="outer")

    observed_height = merged["Height_observed"].replace(missing_value, np.nan)
    calculated_height = merged["Height_calculated"].replace(0, np.nan)

    merged["Height"] = observed_height.combine_first(calculated_height)

    result = (
        merged.loc[:, ["Date", "Height"]]
        .dropna(subset=["Height"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return result


def plot_annual_lake_metrics(
    year: int,
    data: pd.DataFrame,
    thresholds: Mapping[str, float],
    output_dir: str | Path = RESULTS_DIR,
    show: bool = True,
) -> Path:
    """Plot annual RLL, LSA, and LVV series with metric-specific thresholds."""
    required_columns = {"Month", "RLL", "LSA", "LVV"}
    missing_columns = required_columns - set(data.columns)
    missing_thresholds = required_columns.difference({"Month"}) - set(thresholds)

    if missing_columns:
        raise KeyError(f"Missing data columns: {sorted(missing_columns)}")
    if missing_thresholds:
        raise KeyError(f"Missing thresholds: {sorted(missing_thresholds)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax_rll = plt.subplots(figsize=(10, 6))
    ax_lsa = ax_rll.twinx()
    ax_lvv = ax_rll.twinx()
    ax_lvv.spines["right"].set_position(("outward", 60))

    line_rll = ax_rll.plot(data["Month"], data["RLL"], label="RLL")[0]
    line_lsa = ax_lsa.plot(data["Month"], data["LSA"], label="LSA")[0]
    line_lvv = ax_lvv.plot(data["Month"], data["LVV"], label="LVV")[0]

    ax_rll.axhline(thresholds["RLL"], linestyle="--", label="RLL threshold")
    ax_lsa.axhline(thresholds["LSA"], linestyle="--", label="LSA threshold")
    ax_lvv.axhline(thresholds["LVV"], linestyle="--", label="LVV threshold")

    ax_rll.set(
        title=f"Year: {year} – RLL, LSA, and LVV Analysis",
        xlabel="Month",
        ylabel="RLL (m)",
        xticks=range(1, 13),
    )
    ax_rll.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax_lsa.set_ylabel("LSA (km²)")
    ax_lvv.set_ylabel("LVV (Mm³)")
    ax_rll.grid(True)

    handles = (
        ax_rll.get_legend_handles_labels()[0]
        + ax_lsa.get_legend_handles_labels()[0]
        + ax_lvv.get_legend_handles_labels()[0]
    )
    labels = (
        ax_rll.get_legend_handles_labels()[1]
        + ax_lsa.get_legend_handles_labels()[1]
        + ax_lvv.get_legend_handles_labels()[1]
    )
    ax_rll.legend(handles, labels, loc="best")

    output_path = output_dir / f"Lake_Analysis_{year}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return output_path


def _safe_filename(text: str) -> str:
    """Convert a title into a filesystem-safe filename."""
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in text.strip()
    ).strip("_")


def plot_gap_filling_comparison(
    ds_before,
    ds_after,
    start_date="2004",
    band="green",
    scale_max=3000,
    figsize=(12, 6),
    random_seed=None,
):
    """
    Plot a randomly selected Landsat image before and after gap filling.

    Parameters
    ----------
    ds_before : xarray.Dataset
        Original dataset before gap filling.

    ds_after : xarray.Dataset
        Dataset after gap filling.

    start_date : str, default="2004"
        Earliest date to include.

    band : str, default="green"
        Band to display.

    scale_max : float, default=3000
        Maximum value used to scale the displayed band.

    figsize : tuple, default=(12, 6)
        Figure size.

    random_seed : int or None, default=None
        Seed for reproducible random-date selection.

    Returns
    -------
    selected_time
        The randomly selected acquisition date.

    fig
        The generated Matplotlib figure.
    """
    if "time" not in ds_before.dims or "time" not in ds_after.dims:
        raise ValueError("Both datasets must contain a 'time' dimension.")

    if band not in ds_before.data_vars:
        raise KeyError(f"Band '{band}' is not available in ds_before.")

    if band not in ds_after.data_vars:
        raise KeyError(f"Band '{band}' is not available in ds_after.")

    # Filter both datasets
    before_filtered = ds_before.sel(time=slice(start_date, None))
    after_filtered = ds_after.sel(time=slice(start_date, None))

    # Find dates shared by both datasets
    common_times = np.intersect1d(
        before_filtered.time.values,
        after_filtered.time.values,
    )

    if common_times.size == 0:
        raise ValueError(
            f"No common observations were found from {start_date} onwards."
        )

    # Select a random common date
    rng = np.random.default_rng(random_seed)
    selected_time = rng.choice(common_times)

    img_before = before_filtered.sel(time=selected_time)
    img_after = after_filtered.sel(time=selected_time)

    date_label = np.datetime_as_string(selected_time, unit="D")

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_band(
        img_before,
        axes[0],
        f"Before Filling\nLandsat 7 – {date_label}",
        band=band,
        scale_max=scale_max,
    )

    plot_band(
        img_after,
        axes[1],
        f"After Filling\nLandsat 7 – {date_label}",
        band=band,
        scale_max=scale_max,
    )

    fig.suptitle("Landsat 7 Gap-Filling Comparison", fontsize=15)
    fig.tight_layout()
    plt.show()

    return selected_time, fig


def _format_duration(seconds: float) -> str:
    """Convert elapsed seconds into a readable duration."""
    seconds = max(0.0, float(seconds))

    hours, remainder = divmod(int(seconds), 3600)
    minutes, whole_seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {whole_seconds:02d}s"

    if minutes:
        return f"{minutes}m {whole_seconds:02d}s"

    return f"{seconds:.1f}s"


def _estimate_dataset_size_gb(dataset: xr.Dataset) -> float:
    """Estimate the uncompressed dataset size in gigabytes."""
    return sum(
        variable.nbytes
        for variable in dataset.data_vars.values()
    ) / (1024**3)


def _dates_within_tolerance(
    observation_days,
    target_days,
    tolerance_days: int,
) -> np.ndarray:
    """
    Match observation dates to the nearest target date efficiently.

    This avoids creating a large observations × target-dates matrix.
    """
    observation_days = np.asarray(
        observation_days,
        dtype="datetime64[D]",
    )

    target_days = np.sort(
        np.unique(
            np.asarray(
                target_days,
                dtype="datetime64[D]",
            )
        )
    )

    if target_days.size == 0:
        return np.zeros(
            observation_days.size,
            dtype=bool,
        )

    positions = np.searchsorted(
        target_days,
        observation_days,
    )

    left_positions = np.clip(
        positions - 1,
        0,
        target_days.size - 1,
    )

    right_positions = np.clip(
        positions,
        0,
        target_days.size - 1,
    )

    left_difference = np.abs(
        observation_days
        - target_days[left_positions]
    ).astype("timedelta64[D]").astype(int)

    right_difference = np.abs(
        observation_days
        - target_days[right_positions]
    ).astype("timedelta64[D]").astype(int)

    nearest_difference = np.minimum(
        left_difference,
        right_difference,
    )

    return nearest_difference <= tolerance_days


def load_process_satellite_data(
    dc,
    geopolygon_gdf,
    products: Sequence[str],
    time: tuple[str, str],
    measurements: Sequence[str] = ("green", "swir_1"),
    date_list=None,
    tolerance_days: int = 3,
    output_crs: str = "EPSG:6933",
    resolution: tuple[int, int] = (-30, 30),
    dask_chunks: dict | None = None,
    min_gooddata: float = 0.95,
    mask_filters=None,
    group_by: str = "solar_day",
    buffer_degrees: float = 0.0,

    # Mission transition dates
    mission_start_dates: dict[str, str] | None = None,

    # Landsat 7 interpolation
    interpolate_landsat7: bool = True,
    interpolation_dim: str = "y",
    interpolation_method: str = "linear",
    interpolation_other_dim_chunk: int = 128,
    min_missing_fraction: float = 0.01,
    max_missing_fraction: float = 0.35,
    max_gap=None,
    use_nearest_fallback: bool = False,

    # Outputs
    dates_output_path: str | Path = "results/available_dates.txt",
    compute_result: bool = False,
    show_progress: bool = True,
):
    """
    Load Landsat missions separately using one products list and one overall
    date range.

    The overall date range is automatically divided between the supplied
    Landsat products using mission_start_dates.

    Example
    -------
    products = ["ls5_sr", "ls7_sr", "ls8_sr"]
    time = ("1992-01-01", "2024-12-31")

    This becomes:

    ls5_sr: 1992-01-01 to 1999-12-31
    ls7_sr: 2000-01-01 to 2013-12-31
    ls8_sr: 2014-01-01 to 2024-12-31
    """

    total_start = perf_counter()

    if not products:
        raise ValueError("At least one Landsat product must be supplied.")

    if len(time) != 2:
        raise ValueError(
            "time must contain an overall start and end date."
        )

    if tolerance_days < 0:
        raise ValueError(
            "tolerance_days must be zero or greater."
        )

    if interpolation_dim not in {"x", "y"}:
        raise ValueError(
            "interpolation_dim must be either 'x' or 'y'."
        )

    if not (
        0 <= min_missing_fraction
        < max_missing_fraction
        <= 1
    ):
        raise ValueError(
            "Missing fractions must satisfy "
            "0 <= minimum < maximum <= 1."
        )

    # --------------------------------------------------------
    # Default mission start dates
    # --------------------------------------------------------

    if mission_start_dates is None:
        mission_start_dates = {
            "ls5_sr": "1992-01-01",
            "ls7_sr": "2000-01-01",
            "ls8_sr": "2014-01-01",
            "ls9_sr": "2022-01-01",
        }

    unknown_products = [
        product
        for product in products
        if product not in mission_start_dates
    ]

    if unknown_products:
        raise ValueError(
            "No mission start date was supplied for: "
            f"{unknown_products}. Add them to mission_start_dates."
        )

    # --------------------------------------------------------
    # Dask chunks
    # --------------------------------------------------------

    if dask_chunks is None:
        dask_chunks = {
            "time": 1,
            "x": 512,
            "y": 512,
        }
    else:
        dask_chunks = dict(dask_chunks)
        dask_chunks["time"] = 1

    overall_start = pd.Timestamp(time[0])
    overall_end = pd.Timestamp(time[1])

    if overall_start > overall_end:
        raise ValueError(
            "The overall start date occurs after the end date."
        )

    # --------------------------------------------------------
    # Sort products by mission start date
    # --------------------------------------------------------

    ordered_products = sorted(
        products,
        key=lambda product: pd.Timestamp(
            mission_start_dates[product]
        ),
    )

    product_periods = {}

    for index, product in enumerate(ordered_products):
        configured_start = pd.Timestamp(
            mission_start_dates[product]
        )

        product_start = max(
            overall_start,
            configured_start,
        )

        if index < len(ordered_products) - 1:
            next_product = ordered_products[index + 1]

            next_start = pd.Timestamp(
                mission_start_dates[next_product]
            )

            product_end = min(
                overall_end,
                next_start - pd.Timedelta(days=1),
            )
        else:
            product_end = overall_end

        # Skip products outside the requested overall period
        if product_start > product_end:
            continue

        product_periods[product] = (
            product_start.strftime("%Y-%m-%d"),
            product_end.strftime("%Y-%m-%d"),
        )

    if not product_periods:
        raise ValueError(
            "None of the supplied products overlaps the requested period."
        )

    print("Automatically generated product periods:")

    for product, period in product_periods.items():
        print(
            f"  {product:<10} "
            f"{period[0]} to {period[1]}"
        )

    # --------------------------------------------------------
    # Spatial query
    # --------------------------------------------------------

    min_x, min_y, max_x, max_y = (
        geopolygon_gdf.total_bounds
    )

    base_query = {
        "longitude": (
            min_x - buffer_degrees,
            max_x + buffer_degrees,
        ),
        "latitude": (
            min_y - buffer_degrees,
            max_y + buffer_degrees,
        ),
        "resolution": resolution,
        "output_crs": output_crs,
        "group_by": group_by,
    }

    loaded_collections = {}
    datasets_to_merge = []
    gap_statistics = []
    interpolated_dates = []

    # --------------------------------------------------------
    # Load each mission separately
    # --------------------------------------------------------

    for collection_number, (
        product,
        product_time,
    ) in enumerate(product_periods.items(), start=1):

        stage_start = perf_counter()

        print(
            f"\n[{collection_number}/{len(product_periods)}] "
            f"Loading {product}"
        )
        print(
            f"Period: {product_time[0]} to "
            f"{product_time[1]}"
        )

        load_kwargs = {
            "dc": dc,
            "products": [product],
            "measurements": list(measurements),
            "time": product_time,
            "dask_chunks": dask_chunks,
            "min_gooddata": min_gooddata,
            **base_query,
        }

        if mask_filters is not None:
            load_kwargs["mask_filters"] = mask_filters

        dataset = load_ard(**load_kwargs)

        if not isinstance(dataset, xr.Dataset):
            print(
                f"Skipping {product}: load_ard did not "
                "return a Dataset."
            )
            continue

        if dataset.sizes.get("time", 0) == 0:
            print(
                f"Skipping {product}: no observations found."
            )
            continue

        dataset = dataset.sortby("time")

        # Remove duplicate timestamps
        _, unique_indices = np.unique(
            dataset.time.values,
            return_index=True,
        )

        dataset = dataset.isel(
            time=np.sort(unique_indices)
        )

        print(
            f"Observations loaded: "
            f"{dataset.sizes['time']:,}"
        )

        # ----------------------------------------------------
        # Interpolate Landsat 7 only
        # ----------------------------------------------------

        if product == "ls7_sr" and interpolate_landsat7:
            print(
                "Checking Landsat 7 images for missing "
                "scan-line pixels..."
            )

            valid_bands = [
                band
                for band in measurements
                if band in dataset.data_vars
            ]

            if not valid_bands:
                raise ValueError(
                    "No valid Landsat 7 measurements were found."
                )

            # A scan-line pixel is expected to be missing
            # in all requested bands
            detection_data = (
                dataset[valid_bands]
                .to_array(dim="measurement")
            )

            missing_mask = (
                detection_data
                .isnull()
                .all(dim="measurement")
            )

            missing_fraction_lazy = missing_mask.mean(
                dim=("y", "x")
            )

            if show_progress:
                with ProgressBar():
                    missing_fraction = (
                        missing_fraction_lazy.compute()
                    )
            else:
                missing_fraction = (
                    missing_fraction_lazy.compute()
                )

            missing_values = np.asarray(
                missing_fraction.values,
                dtype=float,
            )

            repair_mask = (
                np.isfinite(missing_values)
                & (
                    missing_values
                    >= min_missing_fraction
                )
                & (
                    missing_values
                    <= max_missing_fraction
                )
            )

            repair_indices = np.flatnonzero(
                repair_mask
            )

            unchanged_indices = np.flatnonzero(
                ~repair_mask
            )

            acquisition_times = pd.DatetimeIndex(
                pd.to_datetime(
                    dataset.time.values
                )
            )

            selected_dates = (
                acquisition_times[repair_mask]
                .strftime("%Y-%m-%d")
                .tolist()
            )

            interpolated_dates.extend(
                selected_dates
            )

            gap_statistics.append(
                pd.DataFrame(
                    {
                        "product": product,
                        "time": acquisition_times,
                        "missing_fraction": (
                            missing_values
                        ),
                        "interpolation_selected": (
                            repair_mask
                        ),
                    }
                )
            )

            print(
                f"Images selected for interpolation: "
                f"{repair_indices.size:,}"
            )

            if repair_indices.size > 0:
                scenes_to_fill = dataset.isel(
                    time=repair_indices
                )

                original_descending = (
                    scenes_to_fill[
                        interpolation_dim
                    ].values[0]
                    >
                    scenes_to_fill[
                        interpolation_dim
                    ].values[-1]
                )

                # xarray interpolation requires increasing coordinates
                scenes_to_fill = (
                    scenes_to_fill.sortby(
                        interpolation_dim,
                        ascending=True,
                    )
                )

                other_dim = (
                    "x"
                    if interpolation_dim == "y"
                    else "y"
                )

                scenes_to_fill = (
                    scenes_to_fill.chunk(
                        {
                            "time": 1,
                            interpolation_dim: -1,
                            other_dim: min(
                                interpolation_other_dim_chunk,
                                scenes_to_fill.sizes[
                                    other_dim
                                ],
                            ),
                        }
                    )
                )

                interpolation_kwargs = {
                    "dim": interpolation_dim,
                    "method": interpolation_method,
                }

                if max_gap is not None:
                    interpolation_kwargs[
                        "max_gap"
                    ] = max_gap

                repaired_dataset = (
                    scenes_to_fill.interpolate_na(
                        **interpolation_kwargs
                    )
                )

                if use_nearest_fallback:
                    nearest_kwargs = {
                        "dim": interpolation_dim,
                        "method": "nearest",
                    }

                    if max_gap is not None:
                        nearest_kwargs[
                            "max_gap"
                        ] = max_gap

                    repaired_dataset = (
                        repaired_dataset.interpolate_na(
                            **nearest_kwargs
                        )
                    )

                if original_descending:
                    repaired_dataset = (
                        repaired_dataset.sortby(
                            interpolation_dim,
                            ascending=False,
                        )
                    )

                unchanged_dataset = dataset.isel(
                    time=unchanged_indices
                )

                dataset = xr.concat(
                    [
                        unchanged_dataset,
                        repaired_dataset,
                    ],
                    dim="time",
                    data_vars="all",
                    coords="minimal",
                    compat="override",
                    combine_attrs="override",
                ).sortby("time")

        # Record the source mission
        dataset = dataset.assign_coords(
            satellite_product=(
                "time",
                np.repeat(
                    product,
                    dataset.sizes["time"],
                ),
            )
        )

        loaded_collections[product] = dataset
        datasets_to_merge.append(dataset)

        print(
            f"{product} completed in "
            f"{_format_duration(perf_counter() - stage_start)}"
        )

    if not datasets_to_merge:
        raise ValueError(
            "No Landsat collections were loaded."
        )

    # --------------------------------------------------------
    # Merge all missions
    # --------------------------------------------------------

    print("\nMerging Landsat collections...")

    ds_merged = xr.concat(
        datasets_to_merge,
        dim="time",
        join="outer",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    ).sortby("time")

    _, unique_indices = np.unique(
        ds_merged.time.values,
        return_index=True,
    )

    ds_merged = ds_merged.isel(
        time=np.sort(unique_indices)
    )

    # --------------------------------------------------------
    # Available dates
    # --------------------------------------------------------

    merged_times = pd.DatetimeIndex(
        pd.to_datetime(
            ds_merged.time.values
        )
    )

    merged_calendar_dates = (
        merged_times.normalize()
    )

    available_dates = (
        pd.DatetimeIndex(
            merged_calendar_dates.unique()
        )
        .sort_values()
        .strftime("%Y-%m-%d")
        .tolist()
    )

    output_path = Path(dates_output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(available_dates) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Filter around Date_list
    # --------------------------------------------------------

    if date_list is None:
        ds_filtered = ds_merged
        filtered_dates = available_dates

    else:
        target_dates = pd.DatetimeIndex(
            pd.to_datetime(
                date_list,
                errors="coerce",
            )
        ).dropna().normalize().unique()

        if len(target_dates) == 0:
            raise ValueError(
                "date_list contains no valid dates."
            )

        observation_days = (
            merged_calendar_dates.values.astype(
                "datetime64[D]"
            )
        )

        target_days = (
            target_dates.values.astype(
                "datetime64[D]"
            )
        )

        selection_mask = _dates_within_tolerance(
            observation_days,
            target_days,
            tolerance_days,
        )

        ds_filtered = ds_merged.isel(
            time=np.flatnonzero(
                selection_mask
            )
        )

        filtered_dates = (
            pd.DatetimeIndex(
                merged_calendar_dates[
                    selection_mask
                ].unique()
            )
            .sort_values()
            .strftime("%Y-%m-%d")
            .tolist()
        )

    if compute_result:
        print("\nComputing final dataset...")

        if show_progress:
            with ProgressBar():
                ds_filtered = (
                    ds_filtered.compute()
                )
        else:
            ds_filtered = (
                ds_filtered.compute()
            )

    if gap_statistics:
        gap_statistics = pd.concat(
            gap_statistics,
            ignore_index=True,
        )
    else:
        gap_statistics = pd.DataFrame(
            columns=[
                "product",
                "time",
                "missing_fraction",
                "interpolation_selected",
            ]
        )

    elapsed = perf_counter() - total_start

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(
        f"Merged observations: "
        f"{ds_merged.sizes['time']:,}"
    )
    print(
        f"Filtered observations: "
        f"{ds_filtered.sizes['time']:,}"
    )
    print(
        f"Interpolated Landsat 7 dates: "
        f"{len(interpolated_dates):,}"
    )
    print(
        f"Total runtime: "
        f"{_format_duration(elapsed)}"
    )

    return (
        loaded_collections,
        product_periods,
        ds_merged,
        ds_filtered,
        available_dates,
        filtered_dates,
        interpolated_dates,
        gap_statistics,
    )


def plot_first_last_mndwi(
    mndwi: xr.DataArray,
    number_of_slices: int = 20,
    display_reduction: int = 2,
    col_wrap: int = 5,
    compute: bool = True,
):
    """
    Plot the first and last MNDWI observations efficiently.

    Parameters
    ----------
    mndwi : xr.DataArray
        MNDWI data containing time, y, and x dimensions.

    number_of_slices : int, default=20
        Number of observations selected from each end of the time series.

    display_reduction : int, default=2
        Spatial reduction used only for plotting.

        1 keeps the original resolution.
        2 reduces the number of displayed pixels by approximately four times.
        4 reduces the number of displayed pixels by approximately sixteen times.

    col_wrap : int, default=5
        Number of plot panels per row.

    compute : bool, default=True
        Compute the selected data once before plotting.

    Returns
    -------
    xr.DataArray
        The selected and optionally reduced MNDWI observations.
    """
    if not isinstance(mndwi, xr.DataArray):
        raise TypeError("mndwi must be an xarray.DataArray.")

    if "time" not in mndwi.dims:
        raise ValueError("mndwi must contain a 'time' dimension.")

    if number_of_slices < 1:
        raise ValueError("number_of_slices must be at least 1.")

    if display_reduction < 1:
        raise ValueError("display_reduction must be at least 1.")

    number_of_times = mndwi.sizes["time"]

    if number_of_times == 0:
        raise ValueError("mndwi contains no observations.")

    number_to_select = min(
        number_of_slices,
        number_of_times,
    )

    # Select first and last indices without duplicating observations
    first_indices = np.arange(number_to_select)

    last_indices = np.arange(
        max(number_of_times - number_to_select, 0),
        number_of_times,
    )

    selected_indices = np.unique(
        np.concatenate(
            [first_indices, last_indices]
        )
    )

    selected_mndwi = mndwi.isel(
        time=selected_indices
    )

    # Reduce only the display resolution, not the analytical dataset
    if display_reduction > 1:
        spatial_dimensions = {
            dimension: display_reduction
            for dimension in ("y", "x")
            if dimension in selected_mndwi.dims
        }

        if spatial_dimensions:
            selected_mndwi = selected_mndwi.coarsen(
                spatial_dimensions,
                boundary="trim",
            ).mean()

    print(
        f"Selected {selected_mndwi.sizes['time']} of "
        f"{number_of_times} observations."
    )

    if display_reduction > 1:
        print(
            f"Display resolution reduced by a factor of "
            f"{display_reduction} along each spatial dimension."
        )

    # Run one Dask computation instead of computing panel by panel
    if compute and hasattr(selected_mndwi.data, "compute"):
        print("Loading selected MNDWI images for plotting...")

        with ProgressBar():
            selected_mndwi = selected_mndwi.compute()

        print("Selected images loaded successfully.")

    plot = selected_mndwi.plot(
        col="time",
        col_wrap=col_wrap,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        robust=False,
        size=3.5,
        add_colorbar=True,
    )

    plot.fig.suptitle(
        (
            f"First and Last {number_to_select} "
            "MNDWI Observations"
        ),
        fontsize=15,
        y=1.02,
    )

    return selected_mndwi


def calculate_and_plot_water_area(
    mndwi: xr.DataArray,
    threshold: float = 0.0,
    pixel_length_m: float = 30,
    output_folder: str = "results",
    filename: str = "Area_awi_3dd",
    title: str | None = None,
    compute: bool = True,
):
    """
    Calculate waterbody area from MNDWI and export the result.

    Pixels with MNDWI greater than the threshold are classified as water.

    Parameters
    ----------
    mndwi : xr.DataArray
        MNDWI data with time, y, and x dimensions.

    threshold : float, default=0
        MNDWI threshold used to classify water.

    pixel_length_m : float, default=30
        Pixel length in metres.

    output_folder : str, default="results"
        Folder for PNG and CSV outputs.

    filename : str, default="Area_awi_3dd"
        Output filename without an extension.

    title : str or None
        Optional plot title.

    compute : bool, default=True
        Compute the water-area time series before plotting.

    Returns
    -------
    water_area : xr.DataArray
        Waterbody area in square kilometres for every observation.

    dataframe : pd.DataFrame
        Tabular water-area results.
    """
    start_time = perf_counter()

    if not isinstance(mndwi, xr.DataArray):
        raise TypeError("mndwi must be an xarray.DataArray.")

    required_dimensions = {"time", "y", "x"}
    missing_dimensions = required_dimensions.difference(
        mndwi.dims
    )

    if missing_dimensions:
        raise ValueError(
            f"mndwi is missing dimensions: "
            f"{sorted(missing_dimensions)}"
        )

    if pixel_length_m <= 0:
        raise ValueError(
            "pixel_length_m must be greater than zero."
        )

    output_path = Path(output_folder)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Area represented by one pixel in km²
    area_per_pixel_km2 = (
        pixel_length_m**2 / 1_000_000
    )

    print(
        f"Pixel area: "
        f"{area_per_pixel_km2:.6f} km²"
    )

    # True where the pixel is valid water
    water_mask = (
        mndwi.notnull()
        & (mndwi > threshold)
    )

    # Count water pixels and convert to km²
    water_area = (
        water_mask.sum(
            dim=("y", "x"),
            dtype="int64",
        )
        * area_per_pixel_km2
    )

    water_area = water_area.rename(
        "Water Area"
    )

    water_area.attrs = {
        "long_name": "Observed waterbody area",
        "units": "km²",
        "mndwi_threshold": threshold,
        "pixel_length_metres": pixel_length_m,
        "pixel_area_km2": area_per_pixel_km2,
    }

    # Compute only this small one-dimensional result
    if compute and hasattr(
        water_area.data,
        "compute",
    ):
        print(
            "Calculating water area for "
            f"{water_area.sizes['time']:,} observations..."
        )

        with ProgressBar():
            water_area = water_area.compute()

    # Ensure valid datetime coordinate
    water_area = water_area.assign_coords(
        time=pd.to_datetime(
            water_area.time.values
        )
    )

    if compute and bool(water_area.isnull().any()):
        print(
            "Warning: the calculated time series "
            "contains missing values."
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    if title is None:
        first_date = pd.Timestamp(
            water_area.time.values[0]
        ).strftime("%Y-%m-%d")

        last_date = pd.Timestamp(
            water_area.time.values[-1]
        ).strftime("%Y-%m-%d")

        title = (
            "Observed Seasonal Waterbody Area "
            f"from {first_date} to {last_date}"
        )

    figure, axis = plt.subplots(
        figsize=(18, 4)
    )

    axis.plot(
        water_area.time.values,
        water_area.values,
        marker="o",
        linewidth=1.5,
        markersize=4,
    )

    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Waterbody area (km²)")
    axis.grid(
        True,
        alpha=0.3,
    )

    figure.autofmt_xdate(
        rotation=45
    )

    figure.tight_layout()

    png_path = output_path / f"{filename}.png"

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # --------------------------------------------------------
    # Export CSV
    # --------------------------------------------------------

    dataframe = (
        water_area
        .to_dataframe()
        .reset_index()
    )

    csv_path = output_path / f"{filename}.csv"

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    elapsed = perf_counter() - start_time

    print("\nWater-area calculation completed.")
    print(f"Observations: {len(dataframe):,}")
    print(f"PNG saved to: {png_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Runtime: {elapsed:.1f} seconds")

    return water_area, dataframe
