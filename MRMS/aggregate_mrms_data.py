import xarray as xr
import rioxarray
import os
from datetime import datetime
import dask
from dask.delayed import delayed
from utils import get_file_paths, load_raster
import numpy as np
import rasterio

def aggregate_precipitation_data_dask(directory, output_dir, start_time, end_time, ext='.tif'):
    file_paths = get_file_paths(directory, start_time, end_time, ext = ext)
    file_load_time = datetime.now()
    print(f"Found {len(file_paths)} files in the specified time range.")
    print(f"First file: {file_paths[0]}")
    print(f"Last file: {file_paths[-1]}")
    print(f"Time to find files: {file_load_time - timer_start}")
    # Load all raster files as a dask-backed xarray dataset
    # Using dask.delayed to parallelize the file loading
    delayed_datasets = [delayed(load_raster)(f) for f in file_paths]
    get_datasets_time = datetime.now()
    print(f"Time to create delayed tasks: {get_datasets_time - file_load_time}")
    datasets = dask.compute(*delayed_datasets)  # Compute all delayed tasks to load files
    compute_time = datetime.now()
    print(f"Time to load files: {compute_time - get_datasets_time}")
    dataset = xr.concat(datasets, dim='time')
    concat_time = datetime.now()
    print(f"Time to concatenate files: {concat_time - compute_time}")
    
    # Calculate the running total and maximum intensity using parallel operations
    running_total = dataset.cumsum(dim='time')
    max_60min_intensity = dataset.rolling(time=30, center=True).max()
    
    # Trigger computation and save the results
    running_total_last = running_total.isel(time=-1).compute()  # compute the result
    running_total_time = datetime.now()
    print(f"Time after running total: {running_total_time - concat_time}")
    max_60min_intensity_max = max_60min_intensity.max(dim='time').compute()  # compute the result
    max_int_time = datetime.now()
    print(f"Time after max intensity: {max_int_time - running_total_time}")
    
    running_total_last.rio.to_raster(os.path.join(output_dir, f'running_total_{start_time.strftime("%m%d%y")}_to_{end_time.strftime("%m%d%y")}.tif'))
    max_60min_intensity_max.rio.to_raster(os.path.join(output_dir, f'max_60min_intensity_{start_time.strftime("%m%d%y")}_to_{end_time.strftime("%m%d%y")}.tif'))

def sum_raster_values(raster_files, output_raster):
    """Sum the values of each cell across multiple rasters and save the result to a new raster.

    Args:
        raster_files (list): List of paths to raster files.
        output_raster (str): Path to the output raster file.

    Returns:
        None
    """
    # Open the first raster to initialize the sum_data array and metadata
    with rasterio.open(raster_files[0]) as src0:
        meta = src0.meta.copy()
        # Start with the first raster's data to initialize sum_data
        sum_data = src0.read(1).astype(np.float32)  # Ensure float type for handling potential large values

    # Iterate over the remaining rasters
    for raster in raster_files[1:]:
        with rasterio.open(raster) as src:
            data = src.read(1).astype(np.float32)
            # Use numpy to sum up the data arrays
            sum_data += data

    # Update the metadata to reflect the data type and the number of bands
    meta.update(dtype=rasterio.float32, count=1)

    # Handle no-data values appropriately
    if 'nodata' in meta:
        sum_data[np.isnan(sum_data)] = meta['nodata']
    else:
        meta['nodata'] = -9999  # Define a no-data value if it wasn't set
        sum_data[np.isnan(sum_data)] = meta['nodata']

    # Write the output raster file
    with rasterio.open(output_raster, 'w', **meta) as dst:
        dst.write(sum_data, 1)

def create_max_value_raster(raster_files, output_raster):
    """Create a raster with the maximum value for each cell from a list of raster files.

    Args:
        raster_files (list): List of paths to raster files.
        output_raster (str): Path to the output raster file.

    Returns:
        None
    """
    # Read the first raster to initialize the max_data array and metadata
    with rasterio.open(raster_files[0]) as src0:
        meta = src0.meta.copy()
        max_data = src0.read(1)  # Read only the first band

    # Iterate over the remaining rasters
    for raster in raster_files[1:]:
        with rasterio.open(raster) as src:
            data = src.read(1)  # Read only the first band
            # Update the max_data array with the maximum values
            max_data = np.maximum(max_data, data, where=~np.isnan(data))

    # Update the metadata to reflect the number of bands and data type
    meta.update(dtype=rasterio.float32, count=1)

    # Ensure the no-data values are handled properly
    if 'nodata' in meta:
        max_data[np.isnan(max_data)] = meta['nodata']
    else:
        meta['nodata'] = -9999  # Define a no-data value if it wasn't set
        max_data[np.isnan(max_data)] = meta['nodata']

    # Write the output raster file
    with rasterio.open(output_raster, 'w', **meta) as dst:
        dst.write(max_data, 1)
        
#directory = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\From Megan\2min_rasters\2min_2022"
directory = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\From Megan\2min_rasters"
output_dir = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Precipitation"

start_end_times =[[datetime(2022, 5, 1, 0, 0), datetime(2022, 7, 9, 23, 0)],
                    [datetime(2022, 7, 10, 0, 0), datetime(2022, 10, 31, 23, 0)],
                    [datetime(2023, 5, 1, 0, 0), datetime(2023, 7, 9, 23, 0)],
                    [datetime(2023, 7, 10, 0, 0), datetime(2023, 10, 31, 23, 0)],
                    ]

for start_time, end_time in start_end_times:
    timer_start = datetime.now()
    aggregate_precipitation_data_dask(directory, output_dir, start_time, end_time)
    timer_end = datetime.now()
    #print formatted processing time
    print(f"Processing time: {timer_end - timer_start}")


