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


def process_MRMS_grib_directory(directory, gdf, start_time, end_time):
    
    file_paths = utils.get_file_paths(directory, start_time, end_time, ext = '.gz')
    print(f"Found {len(file_paths)} files in the specified time range.")
    file_load_time = datetime.now()
    print(f"Found {len(file_paths)} files in the specified time range.")

    # Load all raster files as a dask-backed xarray dataset
    # Using dask.delayed to parallelize the file loading
    decompressed_files = [f.replace(".gz", "") for f in file_paths]
    delayed_decompression = []
    for f in file_paths:
        decompressed_file = f.replace(".gz", "")
        delayed_decompression.append(delayed(utils.decompress_grib_gz)(f, decompressed_file))
    delayed_datasets = [delayed(utils.load_raster)(f) for f in decompressed_files]
    get_datasets_time = datetime.now()
    dask.compute(*delayed_decompression)  # Compute to unzip files
    print("Loading files...")
    datasets = dask.compute(*delayed_datasets)  # Compute to load files
    compute_time = datetime.now()
    print(f"Time to load files: {compute_time - get_datasets_time}")
    delayed_datasets = [delayed(utils.clip_xarray_to_gdf)(data, gdf) for data in datasets]
    clipped_datasets = dask.compute(*delayed_datasets)
    print("Concatenating files...")
    dataset = xr.concat(clipped_datasets, dim='time')
    concat_time = datetime.now()
    print(f"Time to concatenate files: {concat_time - compute_time}")
    return dataset




# Path to the compressed GRIB2 file
def main():
    # Path to the directory containing the GRIB2 files
    
    shapefile = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Precipitation\All Watersheds with buffer WGS84.gpkg"
    gdf = gpd.read_file(shapefile)
    input_directory = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Precipitation\MRMS Data\2023"
    start_time = datetime(2023, 5, 1, 0, 0)
    end_time = datetime(2023, 10, 21, 0, 0)

    dataset = process_MRMS_grib_directory(input_directory, gdf, start_time, end_time)
    utils.save_xarray_to_netcdf(dataset, r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Precipitation\MRMS Data\MRMS_QPE_20230501_20231021.nc")

if __name__ == "__main__":
    main()