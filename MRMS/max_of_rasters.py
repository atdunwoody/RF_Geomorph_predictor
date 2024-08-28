import rasterio
import numpy as np

def max_of_rasters(raster_paths, output_path):
    """
    Compute the maximum value at each pixel across multiple overlapping rasters and save the output to a new raster.

    Parameters:
    raster_paths (list of str): List of file paths to the input rasters.
    output_path (str): File path for the output raster.

    """
    # Open the first raster to get the metadata and initialize the max array
    with rasterio.open(raster_paths[0]) as src0:
        meta = src0.meta
        max_array = src0.read(1)

    # Loop over the remaining rasters and update the max_array
    for raster_path in raster_paths[1:]:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            max_array = np.maximum(max_array, data)
    
    # Save the output raster
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(max_array, 1)

def sum_of_rasters(raster_paths, output_path):
    """
    Compute the sum of values at each pixel across multiple overlapping rasters and save the output to a new raster.

    Parameters:
    raster_paths (list of str): List of file paths to the input rasters.
    output_path (str): File path for the output raster.

    """
    # Open the first raster to get the metadata and initialize the sum array
    with rasterio.open(raster_paths[0]) as src0:
        meta = src0.meta
        sum_array = src0.read(1)

    # Loop over the remaining rasters and update the sum_array
    for raster_path in raster_paths[1:]:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            sum_array += data
    
    # Save the output raster
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(sum_array, 1)


raster_paths = [r"Y:\ATD\GIS\MRMS_Data\Summary Data\running_total_050121_to_070921.tif",
            r"Y:\ATD\GIS\MRMS_Data\Summary Data\running_total_050122_to_070922.tif",
            r"Y:\ATD\GIS\MRMS_Data\Summary Data\running_total_050123_to_070923.tif",
            r"Y:\ATD\GIS\MRMS_Data\Summary Data\running_total_071021_to_103121.tif",
            r"Y:\ATD\GIS\MRMS_Data\Summary Data\running_total_071022_to_103122.tif",]
output_path = r"Y:\ATD\GIS\MRMS_Data\Summary Data\running_total_050121_to_070923.tif"
sum_of_rasters(raster_paths, output_path)
