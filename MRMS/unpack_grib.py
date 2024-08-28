import os
import gzip
import shutil
import geopandas as gpd
import xarray as xr
import rioxarray
import pandas as pd
import pygrib
import dask 
from dask import delayed
from datetime import datetime
import glob
import utils
import numpy as np

# Function to process MRMS GRIB files within a specified directory and time range
def process_MRMS_grib_directory(directory, gdf, start_time, end_time):
    
    # Retrieve all file paths within the directory that match the specified time range and extension
    file_paths = utils.get_file_paths(directory, start_time, end_time, ext='.gz')
    print(f"Found {len(file_paths)} files in the specified time range.")

    # Capture the current time for timing purposes
    file_load_time = datetime.now()
    print(f"Found {len(file_paths)} files in the specified time range.")

    # Prepare to load all raster files as a dask-backed xarray dataset
    # Using dask.delayed to parallelize the file decompression and loading processes
    decompressed_files = [f.replace(".gz", "") for f in file_paths]
    delayed_decompression = []
    
    # Prepare a list of delayed tasks for decompressing the .gz files
    for f in file_paths:
        decompressed_file = f.replace(".gz", "")
        delayed_decompression.append(delayed(utils.decompress_grib_gz)(f, decompressed_file))
    
    # Prepare a list of delayed tasks for loading the decompressed raster files
    delayed_datasets = [delayed(utils.load_raster)(f) for f in decompressed_files]
    
    # Capture the time when files are ready to be processed
    get_datasets_time = datetime.now()
    
    # Execute the decompression tasks in parallel
    dask.compute(*delayed_decompression)  # Compute to unzip files
    print("Loading files...")
    
    # Execute the raster loading tasks in parallel
    datasets = dask.compute(*delayed_datasets)  # Compute to load files
    
    # Capture the time after the files are loaded
    compute_time = datetime.now()
    print(f"Time to load files: {compute_time - get_datasets_time}")
    
    # Prepare a list of delayed tasks to clip each dataset to the specified geographic region
    delayed_datasets = [delayed(utils.clip_xarray_to_gdf)(data, gdf) for data in datasets]
    
    # Execute the clipping tasks in parallel
    clipped_datasets = dask.compute(*delayed_datasets)
    print("Concatenating files...")
    
    # Concatenate all the clipped datasets along the time dimension
    dataset = xr.concat(clipped_datasets, dim='time')
    
    # Capture the time after concatenation
    concat_time = datetime.now()
    print(f"Time to concatenate files: {concat_time - compute_time}")
    
    return dataset

# Main function to define file paths, time range, and process the data
def main():
    # Path to the GeoPackage file containing the watershed polygons
    shapefile = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Precipitation\All Watersheds with buffer WGS84.gpkg"
    
    # Load the GeoPackage as a GeoDataFrame
    gdf = gpd.read_file(shapefile)
    
    # Directory containing the GRIB2 files (compressed as .gz)
    input_directory = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Precipitation\MRMS Data\2023"
    
    # Define the start and end time for processing
    start_time = datetime(2023, 5, 1, 0, 0)
    end_time = datetime(2023, 10, 21, 0, 0)

    # Process the GRIB files in the specified directory and time range
    dataset = process_MRMS_grib_directory(input_directory, gdf, start_time, end_time)
    
    # Save the resulting xarray dataset to a NetCDF file
    utils.save_xarray_to_netcdf(dataset, r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Precipitation\MRMS Data\MRMS_QPE_20230501_20231021.nc")

# Entry point of the script
if __name__ == "__main__":
    main()
