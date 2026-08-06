
from pathlib import Path
from urllib.parse import urljoin
import hashlib
import json
import re

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rioxarray
import xarray as xr

from IPython.display import display
from matplotlib.lines import Line2D
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from scipy import ndimage
from tqdm.auto import tqdm

from deafrica_tools.spatial import xr_rasterize, xr_vectorize


# ============================================================
# SETTINGS
# ============================================================

TLU_FACTORS = {
    "cattle": 0.7,
    "buffalo": 0.7,
    "sheep": 0.1,
    "goats": 0.1,
    "pigs": 0.2,
    "chickens": 0.01,
}

GLW4_CODES = {
    "cattle": "CTL",
    "buffalo": "BFL",
    "sheep": "SHP",
    "goats": "GTS",
    "pigs": "PGS",
    "chickens": "CHK",
}

DEFAULT_SPECIES = list(GLW4_CODES)

GLW4_BASE_URL = (
    "https://storage.googleapis.com/"
    "fao-gismgr-glw4-2020-data/"
    "DATA/GLW4-2020/MAPSET/D-DA"
)

OVERTURE_STAC_URL = "https://stac.overturemaps.org/catalog.json"
OVERTURE_S3_ROOT = "s3://overturemaps-us-west-2/release"
OVERTURE_SETTLEMENT_SUBTYPES = (
    "locality",
    "borough",
    "macrohood",
    "neighborhood",
    "microhood",
)


# ============================================================
# PROGRESS
# ============================================================

class WorkflowProgress:
    """Track workflow stage, percentage attempted and percentage left."""

    def __init__(self, stages, enabled=True):
        self.stages = list(stages)
        self.enabled = enabled
        self.completed = 0
        self.bar = tqdm(
            total=len(self.stages),
            desc="Starting workflow",
            unit="stage",
            dynamic_ncols=True,
            disable=not enabled,
        )

    def start(self, label):
        if not self.enabled:
            return

        attempted = 100 * self.completed / len(self.stages)
        self.bar.set_description_str(label)
        self.bar.set_postfix_str(
            f"{attempted:.0f}% attempted | {100-attempted:.0f}% left"
        )

    def finish(self):
        if not self.enabled:
            return

        self.completed += 1
        self.bar.update(1)

        attempted = 100 * self.completed / len(self.stages)
        self.bar.set_postfix_str(
            f"{attempted:.0f}% attempted | {max(0, 100-attempted):.0f}% left"
        )

    def close(self, errors=None, livestock_errors=None):
        if not self.enabled:
            return

        skipped = len(errors or {}) + len(livestock_errors or {})

        if skipped:
            self.bar.set_description_str(
                f"Workflow finished with {skipped} skipped item(s)"
            )
        else:
            self.bar.set_description_str("Workflow complete")

        self.bar.set_postfix_str("100% attempted | 0% left")
        self.bar.close()


# ============================================================
# GENERAL HELPERS
# ============================================================

def country_iso3(country):
    """Convert a country name or ISO3 code to uppercase ISO3."""

    if not isinstance(country, str) or not country.strip():
        raise ValueError("`country` must be a country name or ISO3 code.")

    country = country.strip()

    if len(country) == 3 and country.isalpha():
        return country.upper()

    try:
        import pycountry
        return pycountry.countries.lookup(country).alpha_3
    except ImportError as error:
        raise ImportError(
            "Install pycountry or provide an ISO3 code, e.g. country='ZMB'."
        ) from error
    except LookupError as error:
        raise ValueError(f"Country '{country}' was not recognised.") from error


def raster_crs(data):
    """Return CRS from an ODC or rioxarray object."""

    try:
        if data.odc.crs is not None:
            return data.odc.crs
    except Exception:
        pass

    try:
        if data.rio.crs is not None:
            return data.rio.crs
    except Exception:
        pass

    raise ValueError("Raster CRS is missing.")


def write_crs(data, crs):
    """Write CRS only when it is missing."""
    return data.rio.write_crs(crs) if data.rio.crs is None else data


def scalar(value):
    """Convert xarray, NumPy or Dask scalar to Python float."""

    if hasattr(value, "compute"):
        value = value.compute()

    if hasattr(value, "item"):
        value = value.item()

    return float(value)


def pixel_area_km2(data):
    """Return projected raster pixel area in km²."""

    try:
        res = data.odc.geobox.resolution
        xres, yres = abs(float(res.x)), abs(float(res.y))
    except Exception:
        xres, yres = data.rio.resolution()
        xres, yres = abs(float(xres)), abs(float(yres))

    return xres * yres / 1_000_000


def area_km2(mask):
    """Calculate area occupied by True pixels."""

    count = scalar(mask.fillna(False).astype(bool).sum())
    return count * pixel_area_km2(mask)


def validate_gdf(gdf, name):
    """Validate a GeoDataFrame."""

    if gdf is None:
        raise ValueError(f"`{name}` is None.")
    if gdf.empty:
        raise ValueError(f"`{name}` contains no features.")
    if gdf.crs is None:
        raise ValueError(f"`{name}` has no CRS.")


def rasterize(gdf, template):
    """Rasterize a GeoDataFrame to a Boolean template grid."""

    validate_gdf(gdf, "gdf")
    projected = gdf.to_crs(raster_crs(template))

    return (
        xr_rasterize(gdf=projected, da=template)
        .reindex_like(template, method="nearest")
        .fillna(False)
        .astype(bool)
    )


def to_grid(mask, target, source_crs):
    """Reproject a Boolean mask to a target raster grid."""

    source = (
        write_crs(mask.fillna(False).astype("uint8"), source_crs)
        .rio.write_nodata(0)
    )

    return (
        source.rio.reproject_match(
            target,
            resampling=Resampling.nearest,
        )
        .fillna(0)
        .astype(bool)
    )


def url_exists(url, timeout=60):
    """Return True when a remote file can be reached."""

    response = requests.head(
        url,
        allow_redirects=True,
        timeout=timeout,
    )

    if response.status_code in {403, 405}:
        response = requests.get(
            url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=timeout,
        )

    return response.status_code in {200, 206}


def download_file(
    url,
    destination,
    overwrite=False,
    show_progress=True,
):
    """Stream-download and cache a file."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        tqdm.write(f"Using cached file: {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        with requests.get(url, stream=True, timeout=1200) as response:
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))

            progress = tqdm(
                total=total or None,
                desc=f"Downloading {destination.name}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                disable=not show_progress,
                leave=False,
            )

            with temporary.open("wb") as output:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        progress.update(len(chunk))

            progress.close()

        temporary.replace(destination)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    if not destination.exists() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is empty: {destination}")

    return destination


# ============================================================
# MASKS AND AREA STATISTICS
# ============================================================

def create_masks(
    rnid,
    wetland_boundary,
    livelihood_zone,
    threshold,
):
    """Create wetland, buffer, livelihood and flood masks."""

    validate_gdf(wetland_boundary, "wetland_boundary")
    validate_gdf(livelihood_zone, "livelihood_zone")

    crs = raster_crs(rnid)

    wetland = rasterize(wetland_boundary, rnid)
    livelihood = rasterize(livelihood_zone, rnid)
    flood = (rnid > threshold).fillna(False).astype(bool)
    buffer = livelihood & ~wetland

    return {
        "crs": crs,
        "flood": flood,
        "wetland": wetland,
        "buffer": buffer,
        "livelihood": livelihood,
        "flood_wetland": flood & wetland,
        "flood_buffer": flood & buffer,
        "flood_livelihood": flood & livelihood,
    }


def analysis_zones(masks):
    """Return total-zone and flooded-zone masks."""

    return {
        "Wetland": (
            masks["wetland"],
            masks["flood_wetland"],
        ),
        "Buffer": (
            masks["buffer"],
            masks["flood_buffer"],
        ),
        "Livelihood zone": (
            masks["livelihood"],
            masks["flood_livelihood"],
        ),
    }


def area_statistics(masks):
    """Calculate total and flooded area by zone."""

    rows = []

    for name, (zone, flooded) in analysis_zones(masks).items():
        total = area_km2(zone)
        exposed = area_km2(flooded)

        rows.append({
            "zone": name,
            "total_km2": total,
            "flooded_km2": exposed,
            "pct_flooded": 100 * exposed / total if total > 0 else np.nan,
        })

    return pd.DataFrame(rows).set_index("zone").round(1)


# ============================================================
# WORLDPOP
# ============================================================

def find_tif_urls(value):
    """Extract and normalise WorldPop TIFF URLs."""

    urls = []

    if isinstance(value, str) and ".tif" in value.lower():
        value = value.strip()

        if value.startswith(("http://", "https://")):
            urls.append(value)
        else:
            urls.append(
                urljoin(
                    "https://data.worldpop.org/",
                    value.lstrip("/"),
                )
            )

    elif isinstance(value, list):
        for item in value:
            urls.extend(find_tif_urls(item))

    elif isinstance(value, dict):
        for item in value.values():
            urls.extend(find_tif_urls(item))

    return urls


def worldpop_url(country, year=2020, constrained=True):
    """Find the best WorldPop population-count TIFF URL."""

    iso3 = country_iso3(country)

    response = requests.get(
        "https://www.worldpop.org/rest/data/pop/wpgp",
        params={"iso3": iso3},
        timeout=90,
    )
    response.raise_for_status()

    metadata = response.json()
    records = metadata.get("data", metadata)

    if isinstance(records, dict):
        records = (
            records.get("data")
            or records.get("results")
            or records.get("records")
            or [records]
        )

    if not isinstance(records, list):
        records = [records]

    candidates = []

    for record in records:
        if not isinstance(record, dict):
            continue

        record_year = str(
            record.get("popyear")
            or record.get("year")
            or ""
        )

        if record_year and record_year != str(year):
            continue

        for url in find_tif_urls(record):
            name = Path(url.split("?")[0]).name.lower()
            score = 20 if "_ppp_" in name else 0

            is_constrained = (
                "constrained" in name
                and "unconstrained" not in name
            )

            if constrained:
                score += 100 if is_constrained else 0
                score -= 100 if "unconstrained" in name else 0
            else:
                score += 100 if "unconstrained" in name else 0

            if "unadj" in name:
                score += 10

            candidates.append((score, url))

    if not candidates:
        raise FileNotFoundError(
            f"No WorldPop population-count raster found for {iso3}, {year}."
        )

    return max(candidates, key=lambda item: item[0])[1]


def get_worldpop(
    country,
    year=2020,
    data_folder="Data",
    constrained=True,
    overwrite=False,
    show_progress=True,
):
    """Download or reuse WorldPop data."""

    iso3 = country_iso3(country).lower()
    product = "constrained" if constrained else "unconstrained"

    output = (
        Path(data_folder)
        / "worldpop"
        / f"{iso3}_ppp_{year}_{product}.tif"
    )

    if output.exists() and not overwrite:
        tqdm.write(f"Using cached WorldPop file: {output}")
        return output

    return download_file(
        worldpop_url(country, year, constrained),
        output,
        overwrite=True,
        show_progress=show_progress,
    )


def population_exposure(
    country,
    year,
    masks,
    data_folder="Data",
    constrained=True,
    overwrite=False,
    show_progress=True,
):
    """Calculate population exposure by zone."""

    path = get_worldpop(
        country=country,
        year=year,
        data_folder=data_folder,
        constrained=constrained,
        overwrite=overwrite,
        show_progress=show_progress,
    )

    population = (
        rioxarray.open_rasterio(path, masked=True)
        .squeeze(drop=True)
        .where(lambda data: data >= 0)
    )

    if population.rio.crs is None:
        raise ValueError("WorldPop raster has no CRS.")

    bounds = transform_bounds(
        masks["crs"],
        population.rio.crs,
        *masks["livelihood"].rio.bounds(),
        densify_pts=21,
    )

    population = population.rio.clip_box(
        minx=bounds[0],
        miny=bounds[1],
        maxx=bounds[2],
        maxy=bounds[3],
    )

    flood = to_grid(
        masks["flood_livelihood"],
        population,
        masks["crs"],
    )

    zones = {
        name: to_grid(zone, population, masks["crs"])
        for name, (zone, _) in analysis_zones(masks).items()
    }

    rows = []

    for name, zone in zones.items():
        total = scalar(population.where(zone).sum(skipna=True))
        exposed = scalar(population.where(zone & flood).sum(skipna=True))

        rows.append({
            "zone": name,
            "total_population": total,
            "flood_exposed": exposed,
            "not_exposed": max(total - exposed, 0),
            "pct_exposed": 100 * exposed / total if total > 0 else np.nan,
        })

    table = (
        pd.DataFrame(rows)
        .set_index("zone")
        .round({
            "total_population": 0,
            "flood_exposed": 0,
            "not_exposed": 0,
            "pct_exposed": 1,
        })
    )

    return table, path


# ============================================================
# FAO GLW4 LIVESTOCK
# ============================================================

def glw4_url(species):
    """Return the official FAO GLW4-2020 density raster URL."""

    species = species.lower().strip()

    if species not in GLW4_CODES:
        raise ValueError(
            f"Unsupported species '{species}'. "
            f"Choose from {list(GLW4_CODES)}."
        )

    code = GLW4_CODES[species]

    return (
        f"{GLW4_BASE_URL}/"
        f"GLW4-2020.D-DA.{code}.tif"
    )


def get_glw4_global(
    species,
    data_folder="Data",
    overwrite=False,
    show_progress=True,
):
    """Download or reuse the official FAO GLW4 global raster."""

    code = GLW4_CODES[species]
    output = (
        Path(data_folder)
        / "livestock"
        / "global"
        / f"GLW4-2020.D-DA.{code}.tif"
    )

    if output.exists() and not overwrite:
        tqdm.write(f"Using cached FAO {species} raster: {output}")
        return output

    url = glw4_url(species)

    if not url_exists(url):
        raise FileNotFoundError(
            f"FAO GLW4 raster is unavailable for {species}: {url}"
        )

    return download_file(
        url,
        output,
        overwrite=True,
        show_progress=show_progress,
    )


def get_glw4_aoi(
    species,
    country,
    livelihood_zone,
    data_folder="Data",
    overwrite=False,
    show_progress=True,
):
    """Download, clip and cache one FAO GLW4 AOI raster."""

    iso3 = country_iso3(country).lower()

    output = (
        Path(data_folder)
        / "livestock"
        / "aoi"
        / f"{iso3}_glw4_2020_{species}.tif"
    )

    if output.exists() and not overwrite:
        tqdm.write(f"Using cached AOI {species} raster: {output}")
        return output

    source_path = get_glw4_global(
        species=species,
        data_folder=data_folder,
        overwrite=overwrite,
        show_progress=show_progress,
    )

    source = (
        rioxarray.open_rasterio(source_path, masked=True)
        .squeeze(drop=True)
    )

    if source.rio.crs is None:
        raise ValueError(f"FAO GLW4 raster has no CRS: {source_path}")

    aoi = livelihood_zone.to_crs(source.rio.crs)

    sxmin, symin, sxmax, symax = source.rio.bounds()
    axmin, aymin, axmax, aymax = aoi.total_bounds

    overlaps = not (
        axmax <= sxmin
        or axmin >= sxmax
        or aymax <= symin
        or aymin >= symax
    )

    if not overlaps:
        source.close()
        raise ValueError(
            f"FAO {species} raster does not overlap the AOI."
        )

    clipped = (
        source.rio.clip_box(
            minx=float(axmin),
            miny=float(aymin),
            maxx=float(axmax),
            maxy=float(aymax),
        )
        .rio.clip(
            geometries=aoi.geometry.values,
            crs=aoi.crs,
            drop=True,
            all_touched=True,
        )
        .where(lambda data: data >= 0)
    )

    if clipped.size == 0 or scalar(clipped.notnull().sum()) == 0:
        source.close()
        raise ValueError(
            f"No valid FAO {species} data were found in the AOI."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    clipped.rio.to_raster(
        output,
        compress="DEFLATE",
        tiled=True,
    )

    source.close()
    return output


def livestock_exposure(
    country,
    livelihood_zone,
    masks,
    species=None,
    data_folder="Data",
    overwrite=False,
    show_progress=True,
):
    """Calculate FAO GLW4 livestock exposure and TLU."""

    species = species or DEFAULT_SPECIES

    unsupported = [
        animal for animal in species
        if animal not in GLW4_CODES
    ]

    if unsupported:
        raise ValueError(f"Unsupported species: {unsupported}")

    grid = (
        write_crs(masks["flood"].astype("uint8"), masks["crs"])
        .rio.write_nodata(0)
    )

    cell_area = pixel_area_km2(grid)
    zones = analysis_zones(masks)

    rows = []
    files = {}
    errors = {}

    tlu = {
        name: {"total": 0.0, "exposed": 0.0}
        for name in zones
    }

    bar = tqdm(
        total=len(species),
        desc="FAO GLW4",
        unit="species",
        dynamic_ncols=True,
        disable=not show_progress,
        leave=False,
    )

    for index, animal in enumerate(species, start=1):
        attempted = 100 * (index - 1) / len(species)

        bar.set_description_str(f"FAO GLW4: {animal}")
        bar.set_postfix_str(
            f"{attempted:.0f}% attempted | {100-attempted:.0f}% left"
        )

        try:
            path = get_glw4_aoi(
                species=animal,
                country=country,
                livelihood_zone=livelihood_zone,
                data_folder=data_folder,
                overwrite=overwrite,
                show_progress=show_progress,
            )

            files[animal] = path

            density = (
                rioxarray.open_rasterio(path, masked=True)
                .squeeze(drop=True)
                .rio.reproject_match(
                    grid,
                    resampling=Resampling.bilinear,
                )
                .where(lambda data: data >= 0)
            )

            # GLW4 D-DA values are heads/km².
            animals = density * cell_area
            factor = TLU_FACTORS[animal]
            row = {"species": animal}

            for zone_name, (zone_mask, flood_mask) in zones.items():
                key = zone_name.lower().replace(" ", "_")

                total = scalar(
                    animals.where(zone_mask).sum(skipna=True)
                )

                exposed = scalar(
                    animals.where(flood_mask).sum(skipna=True)
                )

                row[f"{key}_total"] = total
                row[f"{key}_exposed"] = exposed
                row[f"{key}_pct"] = (
                    100 * exposed / total
                    if total > 0
                    else np.nan
                )

                tlu[zone_name]["total"] += total * factor
                tlu[zone_name]["exposed"] += exposed * factor

            rows.append(row)

        except Exception as error:
            errors[animal] = str(error)
            tqdm.write(f"{animal.title()} skipped: {error}")

        finally:
            bar.update(1)

    bar.set_postfix_str("100% attempted | 0% left")
    bar.close()

    species_table = (
        pd.DataFrame(rows).set_index("species").round(1)
        if rows else None
    )

    tlu_rows = []

    for zone, values in tlu.items():
        total = values["total"]
        exposed = values["exposed"]

        tlu_rows.append({
            "zone": zone,
            "total_TLU": total,
            "flood_exposed_TLU": exposed,
            "pct_exposed": (
                100 * exposed / total
                if total > 0
                else np.nan
            ),
        })

    tlu_table = (
        pd.DataFrame(tlu_rows)
        .set_index("zone")
        .round(1)
    )

    return {
        "species": species_table,
        "tlu": tlu_table,
        "files": files,
        "errors": errors,
    }


# ============================================================
# ROADS AND SETTLEMENTS FROM OVERTURE MAPS
# ============================================================

def _load_duckdb_extensions(connection):
    """Load DuckDB spatial and HTTP/S3 extensions."""

    for extension in ("spatial", "httpfs"):
        try:
            connection.execute(f"LOAD {extension};")
        except Exception:
            connection.execute(
                f"INSTALL {extension}; LOAD {extension};"
            )

    connection.execute("SET s3_region='us-west-2';")
    connection.execute("SET s3_access_key_id='';")
    connection.execute("SET s3_secret_access_key='';")


def overture_connection():
    """Create a configured in-memory DuckDB connection."""

    try:
        import duckdb
    except ImportError as error:
        raise ImportError(
            "Install DuckDB with `pip install 'duckdb>=1.1.0'` "
            "before running the Overture stage."
        ) from error

    connection = duckdb.connect(database=":memory:")
    _load_duckdb_extensions(connection)
    return connection


def latest_overture_release(connection=None):
    """Return the release currently identified as latest by Overture STAC."""

    owns_connection = connection is None
    connection = connection or overture_connection()

    try:
        row = connection.execute(
            f"SELECT latest FROM '{OVERTURE_STAC_URL}'"
        ).fetchone()

        if row is None or not row[0]:
            raise RuntimeError(
                "The Overture STAC catalogue did not return a release."
            )

        return str(row[0])

    finally:
        if owns_connection:
            connection.close()


def _from_duckdb_geometry(value):
    """Convert DuckDB WKB output to a Shapely geometry."""

    if value is None:
        return None

    if hasattr(value, "geom_type"):
        return value

    try:
        from shapely import from_wkb
    except ImportError:
        from shapely.wkb import loads as from_wkb

    return from_wkb(bytes(value))


def _frame_to_gdf(frame, columns):
    """Convert a DuckDB DataFrame containing WKB to GeoDataFrame."""

    if frame is None or frame.empty:
        return gpd.GeoDataFrame(
            columns=columns,
            geometry="geometry",
            crs="EPSG:4326",
        )

    frame = frame.copy()
    frame["geometry"] = frame["geometry"].apply(
        _from_duckdb_geometry
    )
    frame = frame[frame["geometry"].notna()].copy()

    return gpd.GeoDataFrame(
        frame,
        geometry="geometry",
        crs="EPSG:4326",
    )


def _aoi_cache_key(aoi_wgs84):
    """Create a stable short identifier for an AOI geometry."""

    geometry = aoi_wgs84.geometry.union_all()

    try:
        geometry = geometry.normalize()
    except Exception:
        pass

    return hashlib.sha1(geometry.wkb).hexdigest()[:12]


def _overture_cache_paths(
    country,
    release,
    aoi_wgs84,
    data_folder="Data",
):
    """Return GeoPackage and metadata paths for an Overture AOI cache."""

    iso3 = country_iso3(country).lower()
    aoi_key = _aoi_cache_key(aoi_wgs84)
    safe_release = re.sub(r"[^0-9A-Za-z_.-]+", "_", release)

    folder = Path(data_folder) / "osm" / "overture"
    stem = f"{iso3}_{safe_release}_{aoi_key}"

    return (
        folder / f"{stem}.gpkg",
        folder / f"{stem}.json",
    )


def _gpkg_layers(path):
    """List GeoPackage layers without requiring a specific IO backend."""

    path = Path(path)

    if not path.exists():
        return set()

    try:
        import pyogrio

        return set(pyogrio.list_layers(path)[:, 0].tolist())
    except Exception:
        try:
            import fiona

            return set(fiona.listlayers(path))
        except Exception:
            return set()


def _empty_roads():
    return gpd.GeoDataFrame(
        columns=["id", "name", "road_class", "subtype", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )


def _empty_settlements():
    return gpd.GeoDataFrame(
        columns=[
            "id",
            "name",
            "subtype",
            "settlement_class",
            "country",
            "population",
            "geometry",
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def _read_overture_cache(gpkg_path, metadata_path):
    """Read cached roads, settlements and metadata."""

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    layers = _gpkg_layers(gpkg_path)

    roads = (
        gpd.read_file(gpkg_path, layer="roads")
        if "roads" in layers
        else _empty_roads()
    )

    settlements = (
        gpd.read_file(gpkg_path, layer="settlements")
        if "settlements" in layers
        else _empty_settlements()
    )

    return {
        "roads": roads,
        "settlements": settlements,
        "release": metadata["release"],
        "cache_file": Path(gpkg_path),
        "metadata_file": metadata_path,
    }


def _write_overture_cache(
    roads,
    settlements,
    release,
    gpkg_path,
    metadata_path,
):
    """Write Overture AOI layers and a small metadata sidecar."""

    gpkg_path = Path(gpkg_path)
    metadata_path = Path(metadata_path)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    gpkg_path.unlink(missing_ok=True)

    wrote_layer = False

    if not roads.empty:
        roads.to_file(
            gpkg_path,
            layer="roads",
            driver="GPKG",
        )
        wrote_layer = True

    if not settlements.empty:
        settlements.to_file(
            gpkg_path,
            layer="settlements",
            driver="GPKG",
            mode="a" if wrote_layer else "w",
        )

    metadata = {
        "release": release,
        "road_features": int(len(roads)),
        "settlement_features": int(len(settlements)),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def query_overture_roads(
    connection,
    release,
    bounds,
):
    """Query road segments intersecting an EPSG:4326 bounding box."""

    xmin, ymin, xmax, ymax = map(float, bounds)

    query = f"""
        SELECT
            id,
            names.primary AS name,
            class AS road_class,
            subtype,
            geometry
        FROM read_parquet(
            '{OVERTURE_S3_ROOT}/{release}/'
            'theme=transportation/type=segment/*',
            hive_partitioning=1
        )
        WHERE subtype = 'road'
          AND bbox.xmin <= {xmax}
          AND bbox.xmax >= {xmin}
          AND bbox.ymin <= {ymax}
          AND bbox.ymax >= {ymin}
    """

    frame = connection.execute(query).fetchdf()

    return _frame_to_gdf(
        frame,
        ["id", "name", "road_class", "subtype", "geometry"],
    )


def query_overture_settlements(
    connection,
    release,
    bounds,
    settlement_subtypes=None,
):
    """Query division points representing settlements and neighbourhoods."""

    xmin, ymin, xmax, ymax = map(float, bounds)
    settlement_subtypes = tuple(
        settlement_subtypes or OVERTURE_SETTLEMENT_SUBTYPES
    )

    allowed = set(OVERTURE_SETTLEMENT_SUBTYPES)
    unsupported = set(settlement_subtypes) - allowed

    if unsupported:
        raise ValueError(
            "Unsupported settlement subtypes: "
            f"{sorted(unsupported)}. Choose from {sorted(allowed)}."
        )

    subtype_sql = ", ".join(
        f"'{value}'" for value in settlement_subtypes
    )

    query = f"""
        SELECT
            id,
            names.primary AS name,
            subtype,
            class AS settlement_class,
            country,
            population,
            geometry
        FROM read_parquet(
            '{OVERTURE_S3_ROOT}/{release}/'
            'theme=divisions/type=division/*',
            hive_partitioning=1
        )
        WHERE subtype IN ({subtype_sql})
          AND bbox.xmin <= {xmax}
          AND bbox.xmax >= {xmin}
          AND bbox.ymin <= {ymax}
          AND bbox.ymax >= {ymin}
    """

    frame = connection.execute(query).fetchdf()

    return _frame_to_gdf(
        frame,
        [
            "id",
            "name",
            "subtype",
            "settlement_class",
            "country",
            "population",
            "geometry",
        ],
    )


def load_overture_aoi(
    country,
    livelihood_zone,
    data_folder="Data",
    release=None,
    settlement_subtypes=None,
    use_cache=True,
    overwrite=False,
):
    """
    Query and cache Overture roads and settlement points for an AOI.

    Set `release` to a fixed Overture release for a reproducible run.
    Leave it as None to use the current release from the STAC catalogue.
    """

    validate_gdf(livelihood_zone, "livelihood_zone")

    aoi_wgs84 = livelihood_zone.to_crs("EPSG:4326")
    bounds = tuple(aoi_wgs84.total_bounds)

    connection = overture_connection()

    try:
        selected_release = (
            str(release)
            if release is not None
            else latest_overture_release(connection)
        )

        gpkg_path, metadata_path = _overture_cache_paths(
            country=country,
            release=selected_release,
            aoi_wgs84=aoi_wgs84,
            data_folder=data_folder,
        )

        if use_cache and not overwrite:
            cached = _read_overture_cache(
                gpkg_path,
                metadata_path,
            )

            if cached is not None:
                tqdm.write(
                    "Using cached Overture data: "
                    f"{metadata_path}"
                )
                return cached

        tqdm.write(
            f"Overture release: {selected_release}"
        )
        tqdm.write("Querying Overture road segments...")

        roads = query_overture_roads(
            connection=connection,
            release=selected_release,
            bounds=bounds,
        )

        tqdm.write(
            f"Road features returned: {len(roads):,}"
        )
        tqdm.write("Querying Overture settlement divisions...")

        settlements = query_overture_settlements(
            connection=connection,
            release=selected_release,
            bounds=bounds,
            settlement_subtypes=settlement_subtypes,
        )

        tqdm.write(
            f"Settlement features returned: {len(settlements):,}"
        )

    finally:
        connection.close()

    # The cloud query uses a bounding box; clip to the exact AOI here.
    if not roads.empty:
        roads = roads[
            roads.geometry.geom_type.isin(
                ["LineString", "MultiLineString"]
            )
        ].copy()
        roads = gpd.clip(roads, aoi_wgs84)
        roads = roads.drop_duplicates(subset=["id"])

    if not settlements.empty:
        settlements = gpd.clip(settlements, aoi_wgs84)
        settlements = settlements.drop_duplicates(subset=["id"])

    if use_cache:
        _write_overture_cache(
            roads=roads,
            settlements=settlements,
            release=selected_release,
            gpkg_path=gpkg_path,
            metadata_path=metadata_path,
        )

    return {
        "roads": roads,
        "settlements": settlements,
        "release": selected_release,
        "cache_file": gpkg_path,
        "metadata_file": metadata_path,
    }


def flood_to_polygons(flood_mask, crs):
    """Convert a Boolean flood raster to polygons."""

    values = flood_mask.fillna(False).values.astype(bool)

    if not values.any():
        return gpd.GeoDataFrame(
            geometry=[],
            crs=crs,
        )

    polygons = xr_vectorize(
        flood_mask.astype("int16"),
        crs=crs,
        mask=values,
    )

    if "attribute" in polygons.columns:
        polygons = polygons[
            polygons["attribute"] == 1
        ]

    return polygons[["geometry"]].to_crs(crs)


def overture_exposure(
    country,
    livelihood_zone,
    masks,
    data_folder="Data",
    release=None,
    settlement_subtypes=None,
    use_cache=True,
    overwrite=False,
):
    """Calculate road and settlement flood exposure from Overture."""

    output_crs = masks["crs"]

    source = load_overture_aoi(
        country=country,
        livelihood_zone=livelihood_zone,
        data_folder=data_folder,
        release=release,
        settlement_subtypes=settlement_subtypes,
        use_cache=use_cache,
        overwrite=overwrite,
    )

    roads = source["roads"].to_crs(output_crs)
    settlements = source["settlements"].to_crs(output_crs)

    flood_polygons = flood_to_polygons(
        masks["flood_livelihood"],
        output_crs,
    )

    if flood_polygons.empty:
        flooded_roads = roads.iloc[0:0].copy()
        flooded_settlements = settlements.iloc[0:0].copy()

    else:
        flooded_roads = (
            gpd.clip(roads, flood_polygons)
            if not roads.empty
            else roads.copy()
        )

        if settlements.empty:
            flooded_settlements = settlements.copy()
        else:
            flooded_settlements = gpd.sjoin(
                settlements,
                flood_polygons,
                how="inner",
                predicate="intersects",
            )
            flooded_settlements = (
                flooded_settlements
                .drop_duplicates(subset=["id"])
            )

    road_total = (
        roads.geometry.length.sum() / 1000
        if not roads.empty
        else 0.0
    )

    road_exposed = (
        flooded_roads.geometry.length.sum() / 1000
        if not flooded_roads.empty
        else 0.0
    )

    settlement_total = len(settlements)
    settlement_exposed = len(flooded_settlements)

    summary = pd.DataFrame(
        {
            "total": [
                road_total,
                settlement_total,
            ],
            "flood_exposed": [
                road_exposed,
                settlement_exposed,
            ],
            "pct_exposed": [
                (
                    100 * road_exposed / road_total
                    if road_total > 0
                    else 0
                ),
                (
                    100 * settlement_exposed / settlement_total
                    if settlement_total > 0
                    else 0
                ),
            ],
            "unit": [
                "km",
                "count",
            ],
        },
        index=[
            "Roads",
            "Settlements",
        ],
    ).round(1)

    return {
        "summary": summary,
        "roads": roads,
        "flooded_roads": flooded_roads,
        "settlements": settlements,
        "flooded_settlements": flooded_settlements,
        "overture_release": source["release"],
        "cache_file": source["cache_file"],
        "metadata_file": source["metadata_file"],
    }


# ============================================================
# COMPLETE WORKFLOW
# ============================================================

def run_exposure_assessment(
    rnid,
    wetland_boundary,
    livelihood_zone,
    threshold,
    country,
    population_year=2020,
    livestock_species=None,
    overture_release=None,
    settlement_subtypes=None,
    use_overture_cache=True,
    data_folder="Data",
    run_population=True,
    run_livestock=True,
    run_osm=True,
    constrained_population=True,
    overwrite=False,
    show_progress=True,
):
    """Run the complete flood-exposure assessment."""

    stages = [
        "Creating analysis masks",
        "Calculating flood areas",
    ]

    if run_population:
        stages.append("Calculating population exposure")
    if run_livestock:
        stages.append("Calculating livestock exposure")
    if run_osm:
        stages.append("Calculating roads and settlements")

    progress = WorkflowProgress(stages, enabled=show_progress)

    results = {
        "masks": None,
        "area": None,
        "population": None,
        "worldpop_file": None,
        "livestock": None,
        "osm": None,
        "errors": {},
    }

    try:
        progress.start("Stage 1: Creating analysis masks")

        masks = create_masks(
            rnid=rnid,
            wetland_boundary=wetland_boundary,
            livelihood_zone=livelihood_zone,
            threshold=threshold,
        )

        results["masks"] = masks
        progress.finish()

        progress.start("Stage 2: Calculating flood areas")
        results["area"] = area_statistics(masks)
        progress.finish()

        stage = 3

        if run_population:
            progress.start(f"Stage {stage}: Population exposure")

            try:
                (
                    results["population"],
                    results["worldpop_file"],
                ) = population_exposure(
                    country=country,
                    year=population_year,
                    masks=masks,
                    data_folder=data_folder,
                    constrained=constrained_population,
                    overwrite=overwrite,
                    show_progress=show_progress,
                )

            except Exception as error:
                results["errors"]["population"] = str(error)
                tqdm.write(f"Population skipped: {error}")

            finally:
                progress.finish()
                stage += 1

        if run_livestock:
            progress.start(f"Stage {stage}: Livestock exposure")

            try:
                results["livestock"] = livestock_exposure(
                    country=country,
                    livelihood_zone=livelihood_zone,
                    masks=masks,
                    species=livestock_species,
                    data_folder=data_folder,
                    overwrite=overwrite,
                    show_progress=show_progress,
                )

            except Exception as error:
                results["errors"]["livestock"] = str(error)
                tqdm.write(f"Livestock skipped: {error}")

            finally:
                progress.finish()
                stage += 1

        if run_osm:
            progress.start(f"Stage {stage}: Roads and settlements")

            try:
                results["osm"] = overture_exposure(
                    country=country,
                    livelihood_zone=livelihood_zone,
                    masks=masks,
                    data_folder=data_folder,
                    release=overture_release,
                    settlement_subtypes=settlement_subtypes,
                    use_cache=use_overture_cache,
                    overwrite=overwrite,
                )

            except Exception as error:
                results["errors"]["osm"] = str(error)
                tqdm.write(f"OSM skipped: {error}")

            finally:
                progress.finish()

    finally:
        livestock_errors = (
            results["livestock"].get("errors", {})
            if results.get("livestock") is not None
            else {}
        )

        progress.close(
            errors=results["errors"],
            livestock_errors=livestock_errors,
        )

    return results


# ============================================================
# DISPLAY
# ============================================================

def display_results(results):
    """Display all available results."""

    if results["area"] is not None:
        print("\nFlood-area statistics")
        display(results["area"])

    if results["population"] is not None:
        print("\nPopulation exposure")
        display(results["population"])

    if results["livestock"] is not None:
        if results["livestock"]["species"] is not None:
            print("\nLivestock exposure by species")
            display(results["livestock"]["species"])

        print("\nLivestock exposure in TLU")
        display(results["livestock"]["tlu"])

        if results["livestock"]["errors"]:
            print("\nLivestock species skipped")

            for species, error in results["livestock"]["errors"].items():
                print(f"- {species}: {error}")

    if results["osm"] is not None:
        print("\nRoad and settlement exposure")
        display(results["osm"]["summary"])

    if results["errors"]:
        print("\nComponents skipped")

        for component, error in results["errors"].items():
            print(f"- {component}: {error}")


# ============================================================
# OPTIONAL CACHE CLEANUP
# ============================================================

def clear_aoi_caches(
    data_folder="Data",
    livestock=True,
    overture=True,
):
    """Remove cached livestock AOI rasters and/or Overture AOI files."""

    root = Path(data_folder)
    folders = []

    if livestock:
        folders.append(root / "livestock" / "aoi")
    if overture:
        folders.append(root / "osm" / "overture")

    removed = 0
    for folder in folders:
        if not folder.exists():
            continue

        for path in folder.rglob("*"):
            if path.is_file():
                path.unlink()
                removed += 1

    print(f"Removed {removed} cached file(s).")


# ============================================================

# ============================================================
# BUFFER EXPOSURE HELPERS
# ============================================================

_SPECIES_PATTERNS = {
    "cattle": ("cattle", "bovine", "bov", "ctl"),
    "buffalo": ("buffalo", "bfl"),
    "sheep": ("sheep", "ovine", "ovin", "shp"),
    "goats": ("goat", "caprine", "capr", "gts"),
    "pigs": ("pig", "swine", "pgs"),
    "chickens": ("chicken", "poultry", "chk"),
}


def _clean_boolean(data):
    """Return a Boolean DataArray with missing values set to False."""
    return data.fillna(False).astype(bool)


def _identify_species(filepath):
    """Identify a livestock species from a GLW raster filename."""
    filename = Path(filepath).name.lower()

    for animal, patterns in _SPECIES_PATTERNS.items():
        for pattern in patterns:
            separated = rf"(^|[._\-]){re.escape(pattern)}([._\-]|$)"
            if pattern in filename or re.search(separated, filename):
                return animal

    return None


def _locate_worldpop_file(worldpop_file=None, search_root="."):
    """Return the requested or best available WorldPop count raster."""
    if worldpop_file is not None:
        path = Path(worldpop_file)
        if not path.exists():
            raise FileNotFoundError(f"WorldPop file does not exist: {path}")
        return path

    candidates = [
        path
        for path in Path(search_root).rglob("*.tif")
        if "ppp" in path.name.lower()
    ]

    if not candidates:
        return None

    def score(path):
        name = path.name.lower()
        value = 0

        if "unconstrained" in name:
            value -= 20
        elif "constrained" in name:
            value += 20

        if "2020" in name:
            value += 10
        if "worldpop" in path.as_posix().lower():
            value += 5

        return value

    return max(candidates, key=score)


def _locate_livestock_files(
    livestock_files=None,
    livestock_folder="Data/livestock/aoi",
    search_root=".",
):
    """Return one available GLW raster per livestock species."""
    if livestock_files is not None:
        if isinstance(livestock_files, dict):
            selected = {
                str(animal).lower(): Path(path)
                for animal, path in livestock_files.items()
            }
        else:
            selected = {}
            for path in livestock_files:
                path = Path(path)
                animal = _identify_species(path)
                if animal is not None:
                    selected[animal] = path

        missing = [str(path) for path in selected.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Livestock files were not found:\n" + "\n".join(missing)
            )
        return selected

    folder = Path(livestock_folder)
    candidates = list(folder.rglob("*.tif")) if folder.exists() else []

    if not candidates:
        candidates = [
            path
            for path in Path(search_root).rglob("*.tif")
            if "glw" in path.name.lower()
        ]

    selected = {}
    for path in candidates:
        animal = _identify_species(path)
        if animal is None:
            continue

        current = selected.get(animal)
        if current is None or (
            "aoi" in path.as_posix().lower()
            and "aoi" not in current.as_posix().lower()
        ):
            selected[animal] = path

    return selected


def _clip_to_template_bounds(raster, template, template_crs):
    """Clip a source raster to template bounds before reprojection."""
    if raster.rio.crs is None:
        raise ValueError("The source raster does not have a CRS.")

    bounds = transform_bounds(
        template_crs,
        raster.rio.crs,
        *template.rio.bounds(),
        densify_pts=21,
    )

    return raster.rio.clip_box(
        minx=bounds[0],
        miny=bounds[1],
        maxx=bounds[2],
        maxy=bounds[3],
    )


# ============================================================
# BUFFER FLOOD-EXPOSURE FUNCTION
# ============================================================

def calculate_buffer_flood_exposure(
    rnid,
    wetland_boundary,
    livelihood_zone,
    threshold,
    worldpop_file=None,
    livestock_folder="Data/livestock/aoi",
    livestock_files=None,
    glw_is_density=True,
    tlu_factors=None,
    minimum_flood_area_km2=0.1,
    search_root=".",
    show_results=True,
):
    """
    Calculate flood exposure inside a wetland and in the surrounding
    livelihood-zone buffer.

    The buffer is defined as livelihood zone minus wetland.

    Returns
    -------
    dict
        Dictionary containing masks, area statistics, population exposure,
        livestock exposure, TLU and source files.
    """
    tlu_factors = dict(TLU_FACTORS if tlu_factors is None else tlu_factors)

    # Reuse the main workflow mask logic rather than rasterizing twice.
    masks = create_masks(
        rnid=rnid,
        wetland_boundary=wetland_boundary,
        livelihood_zone=livelihood_zone,
        threshold=threshold,
    )
    output_crs = masks["crs"]

    rnid_flood = masks["flood"]
    wetland_mask = masks["wetland"]
    livelihood_mask = masks["livelihood"]
    buffer_ring = masks["buffer"]
    wetland_flood = masks["flood_wetland"]
    buffer_flood = masks["flood_buffer"]
    flood_livelihood = masks["flood_livelihood"]

    area_values = {
        "wetland": area_km2(wetland_mask),
        "livelihood": area_km2(livelihood_mask),
        "buffer": area_km2(buffer_ring),
        "wetland_flood": area_km2(wetland_flood),
        "buffer_flood": area_km2(buffer_flood),
        "livelihood_flood": area_km2(flood_livelihood),
    }

    area_results = pd.DataFrame(
        [
            {
                "zone": "Wetland",
                "total_km2": area_values["wetland"],
                "flooded_km2": area_values["wetland_flood"],
            },
            {
                "zone": "Buffer",
                "total_km2": area_values["buffer"],
                "flooded_km2": area_values["buffer_flood"],
            },
            {
                "zone": "Livelihood zone",
                "total_km2": area_values["livelihood"],
                "flooded_km2": area_values["livelihood_flood"],
            },
        ]
    )
    area_results["pct_flooded"] = np.where(
        area_results["total_km2"] > 0,
        100 * area_results["flooded_km2"] / area_results["total_km2"],
        np.nan,
    )
    area_results = area_results.set_index("zone").round(1)

    buffer_raster = (
        write_crs(buffer_ring.astype("uint8"), output_crs)
        .rio.write_nodata(0)
    )
    buffer_flood_raster = (
        write_crs(buffer_flood.astype("uint8"), output_crs)
        .rio.write_nodata(0)
    )

    population_results = None
    livestock_results = None
    affected_population = np.nan
    total_tlu = 0.0
    population_file_used = None
    livestock_files_used = {}

    if area_values["buffer_flood"] >= minimum_flood_area_km2:
        selected_worldpop = _locate_worldpop_file(
            worldpop_file=worldpop_file,
            search_root=search_root,
        )

        if selected_worldpop is not None:
            population_file_used = selected_worldpop
            population_source = rioxarray.open_rasterio(
                selected_worldpop,
                masked=True,
            )

            try:
                population = (
                    population_source.squeeze(drop=True)
                    .where(lambda data: data >= 0)
                )
                population = _clip_to_template_bounds(
                    raster=population,
                    template=buffer_flood_raster,
                    template_crs=output_crs,
                )

                buffer_on_population = to_grid(
                    buffer_ring,
                    population,
                    output_crs,
                )
                flood_on_population = to_grid(
                    buffer_flood,
                    population,
                    output_crs,
                )

                total_buffer_population = scalar(
                    population.where(buffer_on_population).sum(skipna=True)
                )
                affected_population = scalar(
                    population.where(flood_on_population).sum(skipna=True)
                )
            finally:
                population_source.close()

            population_results = pd.DataFrame(
                [
                    {
                        "zone": "Buffer",
                        "total_population": total_buffer_population,
                        "flood_exposed": affected_population,
                        "not_exposed": max(
                            total_buffer_population - affected_population,
                            0,
                        ),
                        "pct_exposed": (
                            100 * affected_population / total_buffer_population
                            if total_buffer_population > 0
                            else np.nan
                        ),
                    }
                ]
            ).set_index("zone")
            population_results = population_results.round(
                {
                    "total_population": 0,
                    "flood_exposed": 0,
                    "not_exposed": 0,
                    "pct_exposed": 1,
                }
            )

        selected_livestock = _locate_livestock_files(
            livestock_files=livestock_files,
            livestock_folder=livestock_folder,
            search_root=search_root,
        )
        livestock_rows = []
        rnid_pixel_area = pixel_area_km2(buffer_flood_raster)

        for animal, filepath in selected_livestock.items():
            if animal not in tlu_factors:
                continue

            livestock_files_used[animal] = filepath
            livestock_source = rioxarray.open_rasterio(filepath, masked=True)

            try:
                livestock_data = (
                    livestock_source.squeeze(drop=True)
                    .where(lambda data: data >= 0)
                )
                livestock_data = _clip_to_template_bounds(
                    raster=livestock_data,
                    template=buffer_flood_raster,
                    template_crs=output_crs,
                )

                resampling_method = (
                    Resampling.average
                    if glw_is_density
                    else getattr(Resampling, "sum", Resampling.nearest)
                )
                livestock_on_rnid = (
                    livestock_data.rio.reproject_match(
                        buffer_flood_raster,
                        resampling=resampling_method,
                    ).where(lambda data: data >= 0)
                )

                livestock_counts = (
                    livestock_on_rnid * rnid_pixel_area
                    if glw_is_density
                    else livestock_on_rnid
                )
                total_animals = scalar(
                    livestock_counts.where(buffer_ring).sum(skipna=True)
                )
                affected_animals = scalar(
                    livestock_counts.where(buffer_flood).sum(skipna=True)
                )
            finally:
                livestock_source.close()

            tlu_factor = tlu_factors[animal]
            total_animal_tlu = total_animals * tlu_factor
            affected_tlu = affected_animals * tlu_factor
            total_tlu += affected_tlu

            livestock_rows.append(
                {
                    "species": animal,
                    "total_in_buffer": total_animals,
                    "flood_exposed": affected_animals,
                    "pct_exposed": (
                        100 * affected_animals / total_animals
                        if total_animals > 0
                        else np.nan
                    ),
                    "TLU_factor": tlu_factor,
                    "total_TLU": total_animal_tlu,
                    "flood_exposed_TLU": affected_tlu,
                    "source_file": str(filepath),
                }
            )

        if livestock_rows:
            livestock_results = (
                pd.DataFrame(livestock_rows)
                .set_index("species")
                .sort_values("flood_exposed", ascending=False)
            )
            round_columns = [
                "total_in_buffer",
                "flood_exposed",
                "total_TLU",
                "flood_exposed_TLU",
            ]
            livestock_results[round_columns] = (
                livestock_results[round_columns].round(1)
            )
            livestock_results["pct_exposed"] = (
                livestock_results["pct_exposed"].round(1)
            )

    summary_rows = [
        {
            "Indicator": "Buffer ring (livelihood zone minus wetland)",
            "Value": area_values["buffer"],
            "Unit": "km²",
        },
        {
            "Indicator": "Flooded area inside wetland",
            "Value": area_values["wetland_flood"],
            "Unit": "km²",
        },
        {
            "Indicator": "Flooded area in buffer",
            "Value": area_values["buffer_flood"],
            "Unit": "km²",
        },
        {
            "Indicator": "Flooded area in livelihood zone",
            "Value": area_values["livelihood_flood"],
            "Unit": "km²",
        },
    ]

    if np.isfinite(affected_population):
        summary_rows.append(
            {
                "Indicator": "People affected by flooding in the buffer",
                "Value": affected_population,
                "Unit": "people",
            }
        )

    if livestock_results is not None:
        summary_rows.append(
            {
                "Indicator": "Livestock affected by flooding in the buffer",
                "Value": total_tlu,
                "Unit": "TLU",
            }
        )

    exposure_summary = pd.DataFrame(summary_rows)
    exposure_summary["Value"] = exposure_summary["Value"].round(1)

    outputs = {
        "masks": {
            "rnid_flood": rnid_flood,
            "wetland": wetland_mask,
            "livelihood": livelihood_mask,
            "buffer": buffer_ring,
            "wetland_flood": wetland_flood,
            "buffer_flood": buffer_flood,
            "livelihood_flood": flood_livelihood,
        },
        "area": area_results,
        "population": population_results,
        "livestock": livestock_results,
        "population_affected": affected_population,
        "total_TLU": total_tlu,
        "summary": exposure_summary,
        "source_files": {
            "worldpop": population_file_used,
            "livestock": livestock_files_used,
        },
    }

    if show_results:
        print("\nArea and flood statistics")
        display(area_results)

        if area_values["buffer_flood"] < minimum_flood_area_km2:
            print(
                "\nFlooding does not extend appreciably "
                "beyond the wetland boundary."
            )
        else:
            if population_results is not None:
                print("\nPopulation exposure in the buffer")
                display(population_results)
            else:
                print(
                    "\nWorldPop raster was not found. "
                    "Population exposure was skipped."
                )

            if livestock_results is not None:
                print("\nLivestock exposure in the buffer")
                display(
                    livestock_results[
                        [
                            "total_in_buffer",
                            "flood_exposed",
                            "pct_exposed",
                            "TLU_factor",
                            "flood_exposed_TLU",
                        ]
                    ]
                )
                print(f"Total flood-exposed livestock: {total_tlu:,.1f} TLU")
            else:
                print("\nNo suitable GLW livestock rasters were found.")

        print("\nExposure summary")
        display(exposure_summary)

    return outputs


def export_results(
    flood_rnid,
    flood_cdat=None,
    flood_ndfi=None,
    wofs_freq=None,
    annual_wet_frequency=None,
    target_year=None,
    output_crs=None,
    wetland_boundary=None,
    livelihood_zone=None,
    biodiversity_area=None,
    results=None,
    exposure_lc=None,
    comparison=None,
    pa_stats=None,
    output_folder="kafue_outputs",
    figure_dpi=150,
    show_figure=True,
    crop_to="wetland",
    map_padding=0.03,
    wetland_color="black",
    livelihood_color="#d62728",
    livelihood_label="Livelihood-zone boundary",
    flood_color="#1f6feb",
    park_name_column="NAME",
    park_colors=None,
):
    """
    Export Kafue flood rasters, CSV tables, and a clean flood-extent map.

    Non-flood and missing flood pixels are transparent. Wetland, livelihood-
    zone, and protected-area boundaries can be overlaid on the map.
    """

    if flood_rnid is None:
        raise ValueError("`flood_rnid` cannot be None.")

    if not isinstance(flood_rnid, xr.DataArray):
        raise TypeError(
            "`flood_rnid` must be an xarray.DataArray, "
            f"not {type(flood_rnid).__name__}."
        )

    output_folder = Path(output_folder)
    raster_folder = output_folder / "rasters"
    table_folder = output_folder / "tables"
    figure_folder = output_folder / "figures"

    for folder in (raster_folder, table_folder, figure_folder):
        folder.mkdir(parents=True, exist_ok=True)

    exported = {
        "rasters": {},
        "tables": {},
        "figures": {},
        "skipped": {},
    }

    def infer_raster_crs(data):
        try:
            if data.rio.crs is not None:
                return data.rio.crs
        except Exception:
            pass

        try:
            if data.odc.crs is not None:
                return data.odc.crs
        except Exception:
            pass

        return None

    resolved_crs = output_crs or infer_raster_crs(flood_rnid)

    if resolved_crs is None:
        raise ValueError(
            "`output_crs` was not supplied and no CRS was found "
            "on `flood_rnid`."
        )

    def as_geodataframe(data, name):
        if data is None:
            return None

        if isinstance(data, gpd.GeoDataFrame):
            frame = data.copy()
        elif isinstance(data, gpd.GeoSeries):
            frame = gpd.GeoDataFrame(
                geometry=data.copy(),
                crs=data.crs,
            )
        else:
            raise TypeError(
                f"`{name}` must be a GeoDataFrame or GeoSeries, "
                f"not {type(data).__name__}."
            )

        if frame.crs is None:
            raise ValueError(f"`{name}` has no CRS.")

        return frame

    def remove_invalid_metadata(data):
        data = data.copy()

        for key in ("_FillValue", "missing_value", "nodata"):
            data.attrs.pop(key, None)
            data.encoding.pop(key, None)

        return data

    def prepare_binary_raster(data):
        if not isinstance(data, xr.DataArray):
            raise TypeError(
                "Binary raster inputs must be xarray.DataArray objects."
            )

        raster = remove_invalid_metadata(
            data.squeeze(drop=True)
        )

        raster = xr.where(
            np.isfinite(raster) & (raster > 0),
            1,
            0,
        ).astype("uint8")

        raster = raster.rio.write_crs(
            resolved_crs,
            inplace=False,
        )

        raster = raster.rio.write_nodata(
            0,
            encoded=True,
            inplace=False,
        )

        return raster

    def prepare_continuous_raster(data, nodata=-9999.0):
        if not isinstance(data, xr.DataArray):
            raise TypeError(
                "Continuous raster inputs must be xarray.DataArray objects."
            )

        raster = remove_invalid_metadata(
            data.squeeze(drop=True)
        )

        raster = xr.where(
            np.isfinite(raster),
            raster,
            nodata,
        ).astype("float32")

        raster = raster.rio.write_crs(
            resolved_crs,
            inplace=False,
        )

        raster = raster.rio.write_nodata(
            nodata,
            encoded=True,
            inplace=False,
        )

        return raster

    def export_raster(data, filename, categorical):
        if data is None:
            exported["skipped"][filename] = "Raster is None."
            return None

        output_path = raster_folder / filename

        try:
            raster = (
                prepare_binary_raster(data)
                if categorical
                else prepare_continuous_raster(data)
            )

            raster.rio.to_raster(
                output_path,
                compress="DEFLATE",
                tiled=True,
                BIGTIFF="IF_SAFER",
            )

            exported["rasters"][output_path.stem] = output_path
            print(f"Wrote raster: {output_path}")
            return output_path

        except Exception as error:
            exported["skipped"][filename] = str(error)
            print(
                f"Raster skipped: {filename}\n"
                f"Reason: {error}"
            )
            return None

    def safe_name(value):
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_") or "table"

    def export_table(table, name, filename=None):
        if table is None:
            return None

        if isinstance(table, (gpd.GeoDataFrame, gpd.GeoSeries)):
            return None

        if isinstance(table, pd.Series):
            table = table.to_frame()

        if not isinstance(table, pd.DataFrame):
            return None

        filename = filename or f"kafue_{safe_name(name)}.csv"
        output_path = table_folder / filename

        table.to_csv(output_path, index=True)
        exported["tables"][name] = output_path
        print(f"Wrote table: {output_path}")
        return output_path

    def export_nested_tables(data, prefix="results", visited=None):
        if visited is None:
            visited = set()

        if data is None:
            return

        object_id = id(data)

        if object_id in visited:
            return

        visited.add(object_id)

        if isinstance(
            data,
            (
                gpd.GeoDataFrame,
                gpd.GeoSeries,
                xr.DataArray,
                xr.Dataset,
                np.ndarray,
            ),
        ):
            return

        if isinstance(data, pd.Series):
            export_table(data, prefix)
            return

        if isinstance(data, pd.DataFrame):
            export_table(data, prefix)
            return

        if not isinstance(data, dict):
            return

        for key, value in data.items():
            child_prefix = f"{prefix}_{safe_name(key)}"

            if isinstance(
                value,
                (
                    gpd.GeoDataFrame,
                    gpd.GeoSeries,
                    xr.DataArray,
                    xr.Dataset,
                    np.ndarray,
                ),
            ):
                continue

            if isinstance(value, (pd.DataFrame, pd.Series)):
                export_table(value, child_prefix)

            elif isinstance(value, dict):
                export_nested_tables(
                    value,
                    prefix=child_prefix,
                    visited=visited,
                )

    def plot_bounds(frame):
        if frame is None or frame.empty:
            return None

        bounds = np.asarray(frame.total_bounds, dtype="float64")

        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            return None

        return bounds

    year_text = "unknown" if target_year is None else str(target_year)

    raster_specs = [
        (
            flood_rnid,
            f"kafue_flood_rnid_{year_text}.tif",
            True,
        ),
        (
            flood_cdat,
            f"kafue_flood_cdat_{year_text}.tif",
            True,
        ),
        (
            flood_ndfi,
            f"kafue_flood_ndfi_{year_text}.tif",
            True,
        ),
        (
            wofs_freq,
            "kafue_wofs_frequency.tif",
            False,
        ),
        (
            annual_wet_frequency,
            f"kafue_annual_wetfreq_{year_text}.tif",
            False,
        ),
    ]

    for raster, filename, categorical in raster_specs:
        export_raster(
            data=raster,
            filename=filename,
            categorical=categorical,
        )

    export_table(
        exposure_lc,
        "landcover_exposure",
        "kafue_landcover_exposure.csv",
    )

    export_table(
        comparison,
        "s1_wofs_comparison",
        "kafue_s1_wofs_comparison.csv",
    )

    export_table(
        pa_stats,
        "protected_area_exposure",
        "kafue_protected_area_exposure.csv",
    )

    export_nested_tables(
        results,
        prefix="results",
    )

    wetland_plot = as_geodataframe(
        wetland_boundary,
        "wetland_boundary",
    )

    if wetland_plot is not None:
        wetland_plot = wetland_plot.to_crs(
            resolved_crs
        )

    livelihood_plot = as_geodataframe(
        livelihood_zone,
        "livelihood_zone",
    )

    if livelihood_plot is not None:
        livelihood_plot = livelihood_plot.to_crs(
            resolved_crs
        )

    parks_plot = as_geodataframe(
        biodiversity_area,
        "biodiversity_area",
    )

    if parks_plot is not None:
        parks_plot = parks_plot.to_crs(
            resolved_crs
        )

    plot_flood = flood_rnid.squeeze(drop=True)

    plot_flood = xr.where(
        np.isfinite(plot_flood) & (plot_flood > 0),
        1.0,
        np.nan,
    )

    try:
        plot_flood = plot_flood.rio.write_crs(
            resolved_crs,
            inplace=False,
        )
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 6))

    plot_flood.plot.imshow(
        ax=ax,
        cmap=mcolors.ListedColormap([flood_color]),
        vmin=0,
        vmax=1,
        add_colorbar=False,
        interpolation="nearest",
    )

    if livelihood_plot is not None and not livelihood_plot.empty:
        livelihood_plot.boundary.plot(
            ax=ax,
            edgecolor=livelihood_color,
            linewidth=1.4,
            linestyle="--",
            label=livelihood_label,
            zorder=2,
        )

    if wetland_plot is not None and not wetland_plot.empty:
        wetland_plot.boundary.plot(
            ax=ax,
            edgecolor=wetland_color,
            linewidth=1.4,
            label="Wetland boundary",
            zorder=3,
        )

    default_park_colors = {
        "Blue Lagoon": "#ef4b3e",
        "Lochinvar": "#ff8c1a",
    }

    selected_park_colors = (
        default_park_colors
        if park_colors is None
        else dict(park_colors)
    )

    if parks_plot is not None and not parks_plot.empty:

        if park_name_column in parks_plot.columns:
            park_names = (
                parks_plot[park_name_column]
                .fillna("")
                .astype(str)
            )

            plotted_index = set()

            for park_name, park_color in selected_park_colors.items():
                mask = park_names.eq(str(park_name))
                park_subset = parks_plot.loc[mask]

                if not park_subset.empty:
                    park_subset.boundary.plot(
                        ax=ax,
                        edgecolor=park_color,
                        linewidth=1.4,
                        label=str(park_name),
                        zorder=4,
                    )

                    plotted_index.update(
                        park_subset.index.tolist()
                    )

            other_parks = parks_plot.loc[
                ~parks_plot.index.isin(plotted_index)
            ]

            if not other_parks.empty:
                other_parks.boundary.plot(
                    ax=ax,
                    edgecolor="#ef4b3e",
                    linewidth=1.0,
                    label="Other protected area",
                    zorder=3,
                )

        else:
            parks_plot.boundary.plot(
                ax=ax,
                edgecolor="#ef4b3e",
                linewidth=1.2,
                label="Protected area",
                zorder=3,
            )

    crop_bounds = None

    if crop_to == "wetland":
        crop_bounds = plot_bounds(wetland_plot)

    elif crop_to == "livelihood":
        crop_bounds = plot_bounds(livelihood_plot)

    elif crop_to == "biodiversity":
        crop_bounds = plot_bounds(parks_plot)

    elif crop_to == "raster":
        try:
            crop_bounds = np.asarray(
                plot_flood.rio.bounds(),
                dtype="float64",
            )
        except Exception:
            crop_bounds = None

    elif crop_to is None:
        crop_bounds = None

    else:
        raise ValueError(
            "`crop_to` must be 'wetland', 'livelihood', "
            "'biodiversity', 'raster', or None."
        )

    if crop_bounds is not None and np.isfinite(crop_bounds).all():
        xmin, ymin, xmax, ymax = crop_bounds

        x_span = xmax - xmin
        y_span = ymax - ymin

        x_pad = x_span * float(map_padding)
        y_pad = y_span * float(map_padding)

        if x_pad == 0:
            x_pad = 1.0

        if y_pad == 0:
            y_pad = 1.0

        ax.set_xlim(xmin - x_pad, xmax + x_pad)
        ax.set_ylim(ymin - y_pad, ymax + y_pad)

    ax.set_title(
        f"Kafue Flats flood extent, {year_text}\n"
        "RNID, locally calibrated",
        fontsize=14,
    )

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_aspect("equal")

    handles, labels = ax.get_legend_handles_labels()

    if handles:
        unique_items = {}

        for handle, label in zip(handles, labels):
            label_text = str(label)

            if label_text not in unique_items:
                unique_items[label_text] = handle

        ax.legend(
            unique_items.values(),
            unique_items.keys(),
            loc="lower right",
            frameon=True,
        )

    figure_path = (
        figure_folder
        / f"kafue_flood_extent_{year_text}.png"
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=figure_dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    exported["figures"]["flood_extent"] = figure_path
    print(f"Wrote figure: {figure_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    print(
        "\nAll available outputs were saved to:"
        f"\n{output_folder.resolve()}"
    )

    if exported["skipped"]:
        print("\nSkipped outputs:")

        for name, reason in exported["skipped"].items():
            print(f"- {name}: {reason}")

    return exported


def add_boundaries(
    ax,
    wetland_gdf=None,
    parks_gdf=None,
    park_column=None,
    wetland=True,
    parks=True,
    legend=False,
    wetland_color="black",
    wetland_label="Kafue Flats Wetland",
    park_colors=None,
    padding=0.03,
):
    """Add wetland and protected-area boundaries with map-edge padding."""

    handles = []

    # Wetland boundary
    if (
        wetland
        and wetland_gdf is not None
        and not wetland_gdf.empty
    ):
        wetland_gdf.boundary.plot(
            ax=ax,
            edgecolor=wetland_color,
            linewidth=1.3,
            zorder=6,
        )

        handles.append(
            Line2D(
                [0],
                [0],
                color=wetland_color,
                lw=1.3,
                label=wetland_label,
            )
        )

    # Protected-area boundaries
    if (
        parks
        and parks_gdf is not None
        and not parks_gdf.empty
    ):
        if park_column not in parks_gdf.columns:
            raise KeyError(
                f"'{park_column}' was not found. "
                f"Available columns: {parks_gdf.columns.tolist()}"
            )

        for label, subset in parks_gdf.groupby(park_column):
            colour = (
                park_colors.get(label, "#d7301f")
                if park_colors
                else "#d7301f"
            )

            subset.boundary.plot(
                ax=ax,
                edgecolor=colour,
                linewidth=1.1,
                zorder=7,
            )

            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=colour,
                    lw=1.1,
                    label=str(label),
                )
            )

    # Add space between boundaries and plot frame
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xpad = abs(xlim[1] - xlim[0]) * padding
    ypad = abs(ylim[1] - ylim[0]) * padding
    
    ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
    
    # Preserve ascending or descending y coordinates
    if ylim[0] < ylim[1]:
        ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)
    else:
        ax.set_ylim(ylim[0] + ypad, ylim[1] - ypad)

    if legend and handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=7,
            framealpha=0.9,
        )

    return ax

def clean(mask, min_size=5):
    """Remove isolated water blobs smaller than min_size pixels.
    At 30 m, 5 pixels is about 0.45 ha, so narrow channels and small ponds
    below that size are removed - state this when reporting the results."""
    lab, n = ndimage.label(mask.values)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n+1))
        mask.values = mask.values & ~np.isin(lab, np.where(sizes < min_size)[0]+1)
    return mask

def protected_area_flood_stats(
    biodiversity_area,
    target_s1,
    flood,
    name_column="NAME",
):
    """Calculate total and flooded area for each protected area."""

    if biodiversity_area is None or biodiversity_area.empty:
        print("No protected-area shapefile loaded — skipping.")
        return None

    # Match vector CRS to the raster CRS
    parks = biodiversity_area.to_crs(target_s1.odc.crs)

    rows = []

    for label, gdf in parks.groupby(name_column):
        try:
            mask = rasterize(gdf, target_s1)
        except Exception as error:
            print(f"Skipped {label}: {error}")
            continue

        total_area = area_km2(mask)
        flooded_area = area_km2(mask & flood)

        if total_area <= 0:
            continue

        rows.append({
            "protected_area": label,
            "area_km2": round(total_area, 1),
            "flooded_km2": round(flooded_area, 1),
            "pct_flooded": round(
                100 * flooded_area / total_area,
                1,
            ),
        })

    if not rows:
        print("No protected-area statistics could be calculated.")
        return None

    return (
        pd.DataFrame(rows)
        .set_index("protected_area")
        .sort_index()
    )

def worldcover_flood_exposure(
    dc,
    flood,
    template,
    wetland_mask,
    x,
    y,
    output_crs,
    resolution,
):
    """Load ESA WorldCover and calculate flooded area by land-cover class."""

    # Align flood and wetland masks to the analysis grid
    flood = _clean_boolean(
        flood.reindex_like(template, method="nearest")
    )
    wetland_mask = _clean_boolean(
        wetland_mask.reindex_like(template, method="nearest")
    )

    # Try available WorldCover product names
    products = [
        ("esa_worldcover_2021", None),
        ("esa_worldcover_2020", None),
        ("esa_worldcover", "2021"),
    ]

    worldcover = None

    for product, time in products:
        try:
            query = {
                "product": product,
                "x": x,
                "y": y,
                "output_crs": output_crs,
                "resolution": resolution,
                "resampling": "nearest",
                "measurements": ["classification"],
                "dask_chunks": {"x": 2048, "y": 2048},
            }

            if time is not None:
                query["time"] = time

            ds = dc.load(**query).squeeze()

            if (
                "classification" in ds
                and ds["classification"].size > 0
            ):
                worldcover = (
                    ds["classification"]
                    .compute()
                    .reindex_like(template, method="nearest")
                )

                print(f"WorldCover loaded from '{product}'")
                break

        except Exception:
            continue

    if worldcover is None:
        raise RuntimeError("Could not load ESA WorldCover.")

    classes = {
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare/sparse",
        80: "Permanent water",
        90: "Herbaceous wetland",
    }

    rows = []

    for class_value, class_name in classes.items():
        class_mask = (
            (worldcover == class_value)
            & wetland_mask
        )

        total_area = area_km2(class_mask)

        if total_area <= 0:
            continue

        flooded_area = area_km2(
            class_mask & flood
        )

        rows.append({
            "land_cover": class_name,
            "total_km2": round(total_area, 1),
            "flooded_km2": round(flooded_area, 1),
            "pct_flooded": round(
                100 * flooded_area / total_area,
                1,
            ),
        })

    result = (
        pd.DataFrame(rows)
        .set_index("land_cover")
    )

    print(
        f"Flood layer used for exposure: "
        f"{area_km2(flood):,.1f} km²"
    )

    return result

# def area_km2(mask):
#     return float(np.asarray(mask).sum())
