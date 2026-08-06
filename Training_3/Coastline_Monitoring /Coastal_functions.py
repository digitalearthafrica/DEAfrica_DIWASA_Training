# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------

import datacube
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import Image
from matplotlib.colors import ListedColormap

# Add coastlines path if necessary (standard DE Africa path)
# sys.path.append(os.path.abspath('/home/jovyan/deafrica-coastlines'))

# DE Africa and Coastlines specific
from deafrica_tools.datahandling import load_best_available_ds, preprocess_s1
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.spatial import subpixel_contours
from deafrica_tools.plotting import rgb
from eo_tides.eo import pixel_tides
from coastlines.raster import tide_cutoffs
from coastlines.vector import points_on_line, annual_movements, calculate_regressions

def process_coastal_data(dc, lat_range, lon_range, time_range, time_step="1Y", product="ls"):
    """Loads data, applies tidal filtering, and calculates water indices."""
    # 1. Load data
    ds, name = load_best_available_ds(dc, lat_range, lon_range, time_range, time_step, set_product=product)
    
    # 2. Add and filter by tides (using EOT20 model)
    ds["tide_m"] = pixel_tides(ds, model="EOT20", directory="/var/share/tide_models", resample=True)
    low, high = tide_cutoffs(ds, ds["tide_m"], tide_centre=0.0)
    mask = (ds.tide_m >= low) & (ds.tide_m <= high)
    ds = ds.where(mask).sel(time=mask.sum(dim=["x", "y"]) > 0)
    
    # 3. Calculate Water Index or Preprocess SAR
    if product in ["ls", "s2", "ls_s2"]:
        ds = calculate_indices(ds, index="MNDWI", satellite_mission=product if product != "ls_s2" else "s2")
        var = "MNDWI"
    else:
        ds = preprocess_s1(ds, filter_size=2, s1_orbit_filtering=True)
        var = "vh"
        
    return ds, var

def extract_shoreline_statistics(ds, var, time_step="1Y", baseline_year=2015):
    """Generates annual summaries, extracts contours, and calculates movement rates."""
    # Annual composites
    ds_summaries = ds[[var]].resample(time=time_step).median("time").compute()
    ds_summaries["time"] = ds_summaries.time.dt.year
    ds_summaries = ds_summaries.rename(time="year")
    
    # Extract contours
    threshold = 0 if var == "MNDWI" else -20 # Standard thresholds
    contours = subpixel_contours(da=ds_summaries[var], z_values=threshold, dim="year", 
                                 crs=ds_summaries.geobox.crs, min_vertices=15).set_index("year")
    
    # Calculate Movement and Regressions
    points = points_on_line(contours, index=baseline_year, distance=20)
    points = annual_movements(points, contours, ds_summaries, baseline_year=baseline_year, water_index=var)
    points = calculate_regressions(points, contours)
    
    return ds_summaries, contours, points

import rioxarray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import Image
from deafrica_tools.plotting import rgb

import rioxarray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import Image
from deafrica_tools.plotting import rgb

def generate_shoreline_animation(ds_selected, contour_gdf, time_step="1Y", output_file="coastline.gif"):
    """
    Creates a cumulative GIF of shoreline movement over an RGB background.
    Previous years remain visible to show the total extent of coastal change.
    """
    # 1. Prepare Vector Data
    gdf = contour_gdf.reset_index()
    # Ensure years are sorted so the animation builds chronologically
    years = sorted(gdf['year'].unique())
    
    # 2. Prepare Raster Data (RGB)
    print("Computing annual RGB composites for background...")
    ds_rgb = (
        ds_selected[["red", "green", "blue"]]
        .resample(time=time_step)
        .median("time")
        .compute()
    )
    
    # Clip the background to the shoreline extent for a professional "zoomed" look
    try:
        extent = gdf.total_bounds
        ds_rgb = ds_rgb.rio.clip_box(
            minx=extent[0] - 0.001, miny=extent[1] - 0.001, 
            maxx=extent[2] + 0.001, maxy=extent[3] + 0.001
        )
    except:
        print("Warning: Spatial clipping skipped. Using full extent.")

    # 3. Setup Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    palette = sns.color_palette("inferno", len(years)).as_hex()
    
    # Plot the initial background (Static for the whole animation)
    rgb(ds_rgb, index=0, ax=ax)
    ax.axis('off')

    # 4. Define the Cumulative Update Function
    def update(i):
        # We do NOT call ax.clear() here. 
        # This allows each new line to be drawn on top of the old ones.
        
        current_year = years[i]
        ax.set_title(f"Shoreline Evolution: 2015–{current_year}", size=18, weight="bold")
        
        # Plot only the current year's line (previous ones stay on the 'ax')
        current_line = gdf[gdf.year == current_year].plot(
            ax=ax, 
            color=palette[i], 
            linewidth=3, 
            alpha=0.8 # Slight transparency helps see the background
        )
        return current_line

    # 5. Generate and Save
    print(f"Generating cumulative animation for {len(years)} frames...")
    ani = animation.FuncAnimation(fig, update, frames=len(years), interval=1000)

    ani.save(output_file, writer='pillow')
    plt.close(fig)
    
    print(f"Cumulative animation saved to: {output_file}")
    return Image(filename=output_file)
           
#--------------2-------------------

import numpy as np
import pandas as pd
from deafrica_tools.datahandling import load_ard
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.spatial import subpixel_contours
from coastlines.raster import tide_cutoffs
from coastlines.vector import points_on_line, annual_movements
from eo_tides.eo import pixel_tides

def load_and_tide_correct(dc, config, lat_range, lon_range, time_range, tide_model_dir):
    """Loads ARD data and applies EOT20 tidal filtering."""
    ds = load_ard(dc, 
                  products=config['products'], 
                  lat=lat_range, 
                  lon=lon_range, 
                  time=time_range, 
                  measurements=config['measurements'], 
                  resolution=config['res'],
                  output_crs="EPSG:6933", 
                  group_by="solar_day", 
                  min_gooddata=0.1,
                  dask_chunks={"time": 1, "x": 2000, "y": 2000})

    if ds is None or len(ds.time) == 0:
        return None

    # Tidal Correction
    try:
        ds["tide_m"] = pixel_tides(ds, model="EOT20", directory=tide_model_dir, resample=True)
        tide_min, tide_max = tide_cutoffs(ds, ds.tide_m)
        ds = ds.where((ds.tide_m >= tide_min) & (ds.tide_m <= tide_max), drop=True)
    except Exception as e:
        print(f"  Warning: Tidal correction failed ({e})")
        
    return ds

def calculate_water_index(ds, name, config):
    """Calculates MNDWI for optical or dB for SAR data."""
    if config['mission_code']:
        if name == "Sentinel-2" and "swir_2" in ds:
            ds = ds.rename({"swir_2": "swir_1"})
        ds = calculate_indices(ds, index=config['index'], satellite_mission=config['mission_code'])
    else:
        # Sentinel-1 Radar processing
        ds[config['index']] = 10 * np.log10(ds[config['index']].clip(min=0.0001))
    
    return ds, config['index']

def extract_annual_shorelines(ds, var, threshold):
    """Resamples to annual medians and extracts subpixel contours."""
    ds_annual = ds[var].resample(time="1Y").median("time").compute()
    ds_annual = ds_annual.rename({"time": "year"})
    ds_annual['year'] = ds_annual.year.dt.year
    
    contours = subpixel_contours(da=ds_annual, z_values=threshold, dim="year", min_vertices=15)
    return ds_annual, contours

def calculate_relative_change(ds_annual, contours, var, name):
    """Calculates mean distance change relative to the first available year."""
    if contours is None:
        return None
        
    contours = contours.set_index("year")
    base_year = int(contours.index.min())
    
    points = points_on_line(contours, index=base_year, distance=30)
    movements = annual_movements(points, contours, ds_annual.to_dataset(name=var), 
                                 baseline_year=base_year, water_index=var)

    dist_cols = [c for c in movements.columns if c.startswith("dist_")]
    mean_change = movements[dist_cols].mean()
    mean_change.index = [int(c.replace("dist_", "")) for c in mean_change.index]
    
    # Normalize to start at 0
    mean_change = mean_change - mean_change.iloc[0]
    return mean_change.to_frame(name=f"{name}_m")    
      