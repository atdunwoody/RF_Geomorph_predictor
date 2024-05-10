import geopandas as gpd
import rasterio
from osgeo import ogr
from datetime import datetime
import os
import gzip
import shutil
import geopandas as gpd
import xarray as xr
import rioxarray
import dask 
from dask import delayed
from datetime import datetime
import glob
import numpy as np

def open_gpkg(gpkg_path, layer_name):
    """
    Open a layer from a GeoPackage file as a GeoDataFrame.

    Parameters:
    - gpkg_path (str): The file path to the GeoPackage file.
    - layer_name (str): The name of the layer to open from the GeoPackage file.

    Returns:
    - GeoDataFrame: The GeoDataFrame containing the layer from the GeoPackage file.
    """
    return gpd.read_file(gpkg_path, layer=layer_name)

def save_gpkg(gdf, layer_name, out_path, overwrite = False):
    """
    Save a GeoDataFrame to a GeoPackage file.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame to save.
    - out_path (str): The file path to save the GeoPackage file.
    - layer_name (str): The name of the layer to save in the GeoPackage file.
    
    """
    if overwrite:
        gdf.to_file(out_path, layer=layer_name, driver='GPKG', mode='w')
    else:
        gdf.to_file(out_path, layer=layer_name, driver='GPKG')

def reproject_geopackage_layers(gpkg_path, raster_path, output_gpkg_path):
    """
    Reprojects all layers in a GeoPackage to match the CRS of the given raster.

    Parameters:
    gpkg_path (str): Path to the input GeoPackage.
    raster_path (str): Path to the input raster.
    output_gpkg_path (str): Path to the output GeoPackage.
    """
    # Open the raster to read its CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    # Open the GeoPackage
    driver = ogr.GetDriverByName('GPKG')
    input_gpkg = driver.Open(gpkg_path, 0)  # 0 means read-only
    layer_count = input_gpkg.GetLayerCount()

    # Process each layer in the GeoPackage
    for i in range(layer_count):
        layer = input_gpkg.GetLayerByIndex(i)
        layer_name = layer.GetName()

        # Load the layer with GeoPandas
        gdf = gpd.read_file(gpkg_path, layer=layer_name)

        # Reproject the GeoDataFrame
        gdf = gdf.to_crs(raster_crs.to_string())

        # Save the reprojected layer to the new GeoPackage
        mode = 'a' if i > 0 else 'w'  # Append if not the first layer
        gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG", mode=mode)

    # Clean up
    input_gpkg = None


def get_file_paths(directory, start_time, end_time, ext = '.tif'):
    """Retrieve .tif files from directory and subdirectories, sorted and filtered by specified datetime range."""
    def is_within_time_range(file_name):
        # Extract time from the filename assuming it's formatted as '..._YYYYMMDD-HHMMSS.ext'
        time_str = file_name.split('_')[-1][:15]  # The date-time string is in the format YYYYMMDD-HHMMSS
        time = datetime.strptime(time_str, '%Y%m%d-%H%M%S')
        return start_time <= time <= end_time

    # Initialize an empty list to store file paths
    files = []

    # Walk through the directory and subdirectories
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            if file.endswith(ext) and is_within_time_range(file):  # Make sure to adjust the file extension if necessary
                files.append(os.path.join(dirpath, file))

    # Sort files by the extracted time from filenames
    files.sort(key=lambda x: datetime.strptime(x.split('_')[-1][:15], '%Y%m%d-%H%M%S'))
    return files
def compress_grib(input_filepath, output_filepath):
    """
    Compress a GRIB file into a .grib.gz file.

    Args:
    input_filepath (str): The path to the input GRIB file.
    output_filepath (str): The path where the compressed file will be saved.
    """
    with open(input_filepath, 'rb') as f_in:
        with gzip.open(output_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"File compressed and saved as {output_filepath}")

def print_variables_in_xarray(data):
    for var in data.variables:
        print(var)     

def decompress_grib_gz(file_path, output_path):
    """Decompress a .grib.gz file to a .grib file."""
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())

def modify_MRMS_crs(data, crs = "epsg:4326"):
    """
    Modify the CRS of an xarray DataArray containing MRMS data.
    Parameters:
    - data: xarray.DataArray, the MRMS data.
    """
    # Check if longitude adjustments are necessary
    data['longitude'] = ((data['longitude'] + 180) % 360) - 180

    # After adjusting longitudes, it may be necessary to sort them if they are not in increasing order
    data = data.sortby('longitude')

    # Set the CRS after adjusting the longitudes
    data.rio.write_crs(crs, inplace=True)
    return data

def clip_xarray_to_gdf(data, gdf):
    total_bounds = gdf.total_bounds
    clipped = data.rio.clip_box(*total_bounds)
    return clipped

def append_to_xarray(data, new_data):
    """
    Append a new DataArray to an existing DataArray along the time dimension.
    Parameters:
    - data: xarray.DataArray, the existing data.
    - new_data: xarray.DataArray, the new data to append.
    """
    return xr.concat([data, new_data], dim='time')

def load_raster(file):
    """Load a raster file with dask and rioxarray."""
    return rioxarray.open_rasterio(file, chunks={'band': 1, 'x': 1024, 'y': 1024}).squeeze(drop=True)

def save_xarray_to_netcdf(data, output_path):
    """Save an xarray DataArray to a NetCDF file."""
    data.to_netcdf(output_path)
    print(f"Data saved to {output_path}")