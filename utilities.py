import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
import rioxarray
import geopandas as gpd
from shapely.geometry import box
import rioxarray
from dask.diagnostics import ProgressBar


def log(msg, log_file, header =False, print_msg = True):
    """Write a message to a log file."""
    if header:
        with open(log_file, 'w') as f:
            f.write('----------------------------------------------\n')
            f.write(msg + '\n')
        if print_msg:
            print(msg)
    else:            
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
        if print_msg:
            print(msg)

def files_from_folder(folder, ext = None, tag = None):
    """Return a list of files from a folder.
    folder: folder path
    ext: file extension
    """
    if ext is None and tag is None:
        return [os.path.join(folder, f) for f in os.listdir(folder)]
    elif ext is not None and tag is None:
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)]
    elif ext is None and tag is not None:
        return [os.path.join(folder, f) for f in os.listdir(folder) if tag in f]
    else:
        return [os.path.join(folder, f) for f in os.listdir(folder) if tag in f and f.endswith(ext)]


def raster_stats(raster_fn):
    """
    Get comprehensive raster statistics, including the raster's resolution, extent, coordinate system,
    rows, columns, and masked statistics for each band.

    Parameters:
        raster_fn (str): File path to the raster dataset.

    Returns:
        dict: A dictionary containing comprehensive statistics for each band of the raster,
              ignoring no-data values, and including resolution, extent, coordinate system,
              rows, and columns.
    """
    stats = {}
    with rasterio.open(raster_fn) as src:
        # Resolution, rows, columns, extent, and CRS
        resolution = src.res  # Resolution in units per pixel (width, height)
        rows, cols = src.shape  # Number of rows and columns
        extent = src.bounds  # Geographical extent of the raster
        crs = src.crs  # Coordinate reference system

        # Print general raster properties
        print(f"Raster Resolution: {resolution} (width x height in units)")
        print(f"Raster Dimensions: {cols} columns x {rows} rows")
        print(f"Raster Extent: {extent}")
        print(f"Raster CRS: {crs}")

        # Loop through each band in the raster
        for i in range(1, src.count + 1):
            band = src.read(i, masked=True)  # Read with automatic masking of no-data
            nodata = src.nodatavals[i - 1]  # Get the no-data value for the current band

            # Calculate statistics for the current band using the masked array
            stats[f'Band {i}'] = {
                'min': band.min(),
                'max': band.max(),
                'mean': band.mean(),
                'std': band.std(),
                'no_data_value': nodata,
                'crs': crs,
                'rows': rows,
                'cols': cols,
                'extent': extent,
                'resolution': resolution
            }
            # Print statistics for current band, including the no-data value
            print(f"Statistics for Band {i}:")
            print(f"  Min: {stats[f'Band {i}']['min']}")
            print(f"  Max: {stats[f'Band {i}']['max']}")
            print(f"  Mean: {stats[f'Band {i}']['mean']}")
            print(f"  Standard Deviation: {stats[f'Band {i}']['std']}")
            print(f"  No Data Value: {nodata}")

    return stats

def align_raster_crs(raster_fn1, raster_fn2, output_fn = None):
    """
    Reprojects the second raster to match the coordinate system of the first raster.

    Parameters:
        raster_fn1 (str): File path to the reference raster dataset.
        raster_fn2 (str): File path to the raster dataset to be reprojected.
        output_fn (str): File path where the reprojected raster will be saved.

    Returns:
        None: The function saves the reprojected raster to the specified output file path.
    """
    # Load the rasters using rioxarray
    raster1 = rioxarray.open_rasterio(raster_fn1)
    raster2 = rioxarray.open_rasterio(raster_fn2)

    # Fetch the CRS of the first raster
    crs_raster1 = raster1.rio.crs
    crs_raster2 = raster2.rio.crs
    print(f"Modifying CRS of {raster_fn2} to match {raster_fn1}...")
    # Reproject the second raster to match the CRS of the first raster
    #check if the crs are not the same
    if crs_raster1 != crs_raster2:
        raster2_aligned = raster2.rio.reproject(crs_raster1)

    # Save the aligned raster to a new file
    if output_fn is not None:
        raster2_aligned.rio.to_raster(output_fn)
        print(f"Reprojected raster saved to {output_fn}")

    return raster2_aligned

def align_raster_extents(raster_fn1, raster_fn2, method='intersection', output_fn1=None, output_fn2=None):
    """
    Aligns the extents of two rasters either by clipping them to their intersection or extending them to their union.

    Parameters:
        raster_fn1 (str): File path to the first raster dataset.
        raster_fn2 (str): File path to the second raster dataset.
        method (str): Method to align extents, 'intersection' for the intersected area, 'union' for the combined area.
        output_fn1 (str): Optional file path to save the adjusted first raster.
        output_fn2 (str): Optional file path to save the adjusted second raster.

    Returns:
        tuple: A tuple containing the adjusted rioxarray datasets (raster1, raster2).
    """
    # Load the rasters
    raster1 = rioxarray.open_rasterio(raster_fn1)
    raster2 = rioxarray.open_rasterio(raster_fn2)

    # Create GeoDataFrames from raster extents
    geom1 = box(*raster1.rio.bounds())
    geom2 = box(*raster2.rio.bounds())
    gdf1 = gpd.GeoDataFrame({'geometry': [geom1]}, crs=raster1.rio.crs)
    gdf2 = gpd.GeoDataFrame({'geometry': [geom2]}, crs=raster2.rio.crs)

    if method == 'intersection':
        # Compute the intersection of extents
        intersection = gpd.overlay(gdf1, gdf2, how='intersection')
        if intersection.empty:
            raise ValueError("Rasters do not overlap; cannot compute intersection.")
        new_extent = intersection.total_bounds
    elif method == 'union':
        # Compute the union of extents
        union = gpd.overlay(gdf1, gdf2, how='union')
        new_extent = union.total_bounds
    else:
        raise ValueError("Invalid method specified. Use 'intersection' or 'union'.")

    # Clip or extend both rasters to the new extent
    raster1_aligned = raster1.rio.clip_box(*new_extent)
    raster2_aligned = raster2.rio.clip_box(*new_extent)

    # Optionally save the aligned rasters
    if output_fn1:
        raster1_aligned.rio.to_raster(output_fn1)
        print(f"Aligned raster 1 saved to {output_fn1}")
    if output_fn2:
        raster2_aligned.rio.to_raster(output_fn2)
        print(f"Aligned raster 2 saved to {output_fn2}")

    return (raster1_aligned, raster2_aligned)

def resample_raster_to_match(raster_to_resample, reference_raster, match_finer=True, method='nearest'):
    """
    Resample a raster dataset to match the resolution of a reference raster,
    choosing either the finer or coarser resolution.
    
    Parameters:
        raster_to_resample (rioxarray object): The raster to be resampled.
        reference_raster (rioxarray object): The raster whose resolution is used as the target.
        match_finer (bool): If True, match to the finer resolution; otherwise, match to the coarser.
        method (str): Resampling method - options include 'nearest', 'bilinear', 'cubic', etc.
    
    Returns:
        rioxarray object: Resampled raster.
    """
    # Get the resolutions of both rasters
    res_x1, res_y1 = raster_to_resample.rio.resolution()
    res_x2, res_y2 = reference_raster.rio.resolution()

    # Determine the target resolution
    if match_finer:
        target_resolution = (min(abs(res_x1), abs(res_x2)), min(abs(res_y1), abs(res_y2)))
    else:
        target_resolution = (max(abs(res_x1), abs(res_x2)), max(abs(res_y1), abs(res_y2)))

    return resample_raster_to_resolution(raster_to_resample, target_resolution, method)

def resample_raster_to_resolution(raster, target_resolution, method='nearest'):
    """
    Resample a raster dataset to a specified resolution.
    """
    if raster.rio.encoded_nodata is None:
        raster.rio.write_nodata(-9999, inplace=True)
    
    if 'chunks' not in raster.encoding:
        raster = raster.chunk({'band': 1, 'x': 2048, 'y': 2048})

    width = (raster.rio.bounds()[2] - raster.rio.bounds()[0]) / target_resolution[0]
    height = (raster.rio.bounds()[3] - raster.rio.bounds()[1]) / target_resolution[1]

    resampled_raster = raster.rio.reproject(
        raster.rio.crs,
        shape=(int(height), int(width)),
        resampling_method=method
    )

    return resampled_raster

def match_rasters(raster1_fn, raster2_fn, output_fn=None, match_finer=True, method='nearest'):
    # Example usage:
    # Open two raster files using rioxarray
    raster1 = rioxarray.open_rasterio(raster1_fn, chunks=True)
    raster2 = rioxarray.open_rasterio(raster2_fn, chunks=True)

    # Resample raster1 to match the resolution of raster2 (finer resolution)
    resampled_raster = resample_raster_to_match(raster1, raster2, match_finer = match_finer, method=method)

    if output_fn is not None:
        # Save the resampled raster to a new file
        with ProgressBar():
            resampled_raster.rio.to_raster(output_fn)
    return resampled_raster