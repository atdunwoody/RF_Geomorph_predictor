import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling
import shapely
from shapely.geometry import box
import json
import numpy as np
from osgeo import gdal
import os
import numpy as np
import rioxarray
import geopandas as gpd
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

def resample_raster_to_match(raster_to_resample, reference_raster, match_res=True, method=Resampling.nearest):
    """
    Resample a raster dataset to match the resolution of a reference raster,
    choosing either the finer or coarser resolution.
    
    Parameters:
        raster_to_resample (rioxarray object): The raster to be resampled.
        reference_raster (rioxarray object): The raster whose resolution is used as the target.
        match_res (bool, int, float): If True, match to the finer resolution; false, match to the coarser, or specify resolution
        method (str): Resampling method - options include 'nearest', 'bilinear', 'cubic', etc.
    
    Returns:
        rioxarray object: Resampled raster.
    """
    # Get the resolutions of both rasters
    res_x1, res_y1 = raster_to_resample.rio.resolution()
    res_x2, res_y2 = reference_raster.rio.resolution()

    # Determine the target resolution
    if type(match_res) == int or type(match_res) == float:
        target_resolution = (abs(match_res), abs(match_res))
        
    elif match_res and type(match_res) == bool:
        target_resolution = (min(abs(res_x1), abs(res_x2)), min(abs(res_y1), abs(res_y2)))
    
    elif not match_res and type(match_res) == bool:
        target_resolution = (max(abs(res_x1), abs(res_x2)), max(abs(res_y1), abs(res_y2)))

    else:
        raise ValueError("Invalid match_res argument. Use True, False, or a specific resolution.")
    
    return resample_raster_to_resolution(raster_to_resample, target_resolution, method)

def resample_raster_to_resolution(raster, target_resolution, method= Resampling.nearest):
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

def match_rasters(source_raster_fn, ref_raster_fn, output_fn=None, match_res=True, method=Resampling.nearest):
    raster1 = rioxarray.open_rasterio(source_raster_fn, chunks=True)
    raster2 = rioxarray.open_rasterio(ref_raster_fn, chunks=True)

    # Resample raster1 to match the resolution of raster2 (finer resolution)
    resampled_raster = resample_raster_to_match(raster1, raster2, match_res = match_res, method=method)

    if output_fn is not None:
        # Save the resampled raster to a new file
        with ProgressBar():
            resampled_raster.rio.to_raster(output_fn)
    return resampled_raster

def save_dem(data_array, template_raster_path, output_path, data_type=gdal.GDT_Float32):
    """
    Save a DEM-derived raster using a template raster for georeferencing and projection.

    Parameters:
    - data_array: numpy array containing the raster data to be saved.
    - template_raster_path: path to the raster file to use as a template for geospatial metadata.
    - output_path: path where the new raster will be saved.
    - data_type: GDAL data type of the output raster.
    """
    # Open the template raster to read geospatial metadata
    template_raster = gdal.Open(template_raster_path)
    driver = gdal.GetDriverByName('GTiff')
    
    # Create a new raster for output
    out_raster = driver.Create(output_path, template_raster.RasterXSize, template_raster.RasterYSize, 1, data_type)
    out_raster.SetGeoTransform(template_raster.GetGeoTransform())
    out_raster.SetProjection(template_raster.GetProjection())

    # Write the data array to the raster band
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(data_array)
    out_band.SetNoDataValue(np.nan)  # Assuming NaN is used for no-data values; adjust as necessary
    out_band.FlushCache()

    # Close datasets
    template_raster = None
    out_raster = None

def get_raster_extent(raster_path):
    """ Returns the bounding box (extent) of the raster. """
    with rasterio.open(raster_path) as src:
        bbox = src.bounds
    return bbox

def clip_raster_by_extent(raster_path, output_path, minx, miny, maxx, maxy):
    """ Clips a raster by a given bounding box and saves the result. """
    bbox = box(minx, miny, maxx, maxy)
    geo = json.loads(json.dumps(shapely.geometry.mapping(bbox)))
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, [geo], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

def clip_raster_by_another_raster_extent(base_raster_path, clip_raster_path, output_path):
    """ Clips one raster by the extent of another raster. """
    print(f"Clipping {base_raster_path} by the extent of {clip_raster_path}")
    extent = get_raster_extent(clip_raster_path)
    clip_raster_by_extent(base_raster_path, output_path, *extent)

def change_raster_crs(input_raster_path, output_raster_path, new_crs, resampling_mode = Resampling.bilinear):
    """ Changes the CRS of a raster and saves the new raster. """
    with rasterio.open(input_raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, new_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': new_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(output_raster_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=new_crs,
                    resampling=resampling_mode)

def merge_multiband_rasters(raster_paths, output_path):
    """
    Merge multiple multiband raster files into one output file.

    Parameters:
    - raster_paths: list of file paths to the raster files.
    - output_path: file path where the merged raster will be saved.
    """
    # Open all rasters
    src_files = [rasterio.open(rp) for rp in raster_paths]

    # Merge function from rasterio
    mosaic, out_trans = merge(src_files)

    # Copy the metadata of the first raster
    out_meta = src_files[0].meta.copy()

    # Update the metadata to reflect the number of layers
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0]  # Number of bands
    })

    # Write the mosaic raster to disk
    with rasterio.open(output_path, "w", **out_meta) as dest:
        for i in range(1, mosaic.shape[0] + 1):
            dest.write(mosaic[i - 1, :, :], i)

    # Close all opened rasters
    for src in src_files:
        src.close()

def apply_math_to_raster(input_raster_path, output_raster_path, math_func, no_data_value = None):
    """
    Applies a mathematical operation to a raster based on a user-defined lambda function.

    Parameters:
    - input_raster_path (str): Path to the input raster file. This raster provides the values that will be transformed.
    - output_raster_path (str): Path where the transformed raster will be saved.
    - math_func (callable): A function that takes a numpy array (input raster data) and returns a numpy array after applying the mathematical transformation.

    Examples of math_func:
    - Lambda x: x + 10: Adds 10 to each pixel value in the raster.
    - Lambda x: x * 2: Multiplies each pixel value in the raster by 2.
    - Lambda x: np.log(x + 1): Applies the natural logarithm function to each pixel value in the raster after adding 1 to avoid log(0).

    Returns:
    - None: The function saves the transformed raster to the specified output path and does not return any value.
    """
    # Open the input raster
    with rasterio.open(input_raster_path) as input_rast:
        data = input_rast.read(1)  # Read the first band
        profile = input_rast.profile  # Get the metadata to write back the output
    #filter out no data values
    if no_data_value is None:
        data = np.where(data == input_rast.nodata, np.nan, data)
    else:
        data = np.where(data == no_data_value, np.nan, data)
    # Apply the mathematical operation using the provided lambda function
    transformed_data = math_func(data)

    # Save the transformed raster
    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(transformed_data, 1)

def ensure_matching_rasters(threshold_raster_path, target_raster_path):
    """
    Ensure that two rasters match in resolution, shape, and CRS.
    If they don't, resample the threshold raster to match the target raster.
    
    Parameters:
    - threshold_raster_path (str): Path to the threshold raster file to be resampled.
    - target_raster_path (str): Path to the target raster file that will be used as a reference.
    
    Returns:
    - threshold_data_resampled (numpy array): Resampled threshold raster data.
    - target_data (numpy array): Target raster data.
    - target_profile (dict): Profile of the target raster.
    """
    with rasterio.open(threshold_raster_path) as threshold_rast:
        threshold_data = threshold_rast.read(1)
        threshold_transform = threshold_rast.transform
        threshold_crs = threshold_rast.crs

    with rasterio.open(target_raster_path) as target_rast:
        target_data = target_rast.read(1)
        target_transform = target_rast.transform
        target_crs = target_rast.crs
        target_profile = target_rast.profile

    if (threshold_data.shape != target_data.shape) or (threshold_crs != target_crs) or (threshold_transform != target_transform):
        threshold_data_resampled = np.empty(target_data.shape, dtype=target_rast.dtypes[0])
        reproject(
            source=threshold_data,
            destination=threshold_data_resampled,
            src_transform=threshold_transform,
            src_crs=threshold_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
    else:
        threshold_data_resampled = threshold_data

    return threshold_data_resampled, target_data, target_profile

def filter_raster_by_threshold(threshold_raster_path, raster_to_filter_path, output_path, condition_func):
    """
    Filters a raster based on a condition function comparing values between
    a resampled threshold raster and the raster to be filtered.
    
    Parameters:
    - threshold_raster_path (str): Path to the threshold raster file.
    - raster_to_filter_path (str): Path to the raster file that needs filtering.
    - output_path (str): Path where the filtered raster will be saved.
    - condition_func (callable): Function taking two numpy arrays and returning a boolean array.
    """
    threshold_data_resampled, raster_data, profile = ensure_matching_rasters(threshold_raster_path, raster_to_filter_path)
    #filter threshold data for no data values
    threshold_data_resampled = np.where(threshold_data_resampled == profile['nodata'], np.nan, threshold_data_resampled)
    
    # Apply the custom condition function to generate a mask
    mask = condition_func(threshold_data_resampled, raster_data)
    #create second mask for no data values
    mask = np.where(np.isnan(threshold_data_resampled), False, mask)

    print("Mask True count:", np.sum(mask))
    print("Mask False count:", np.sum(~mask))

    # Apply the mask
    masked_data = np.where(mask, raster_data, np.nan)  # Set unmet conditions to NaN

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the filtered raster
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(masked_data, 1)


def main():
    raster_files = [
    r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Aligned DoDs\DoD 070923 SfM\LM2_070923_5cm_veg_masked_QGIS_LM2_081222_5cm_veg_masked_QGIS_nuth_x-0.00_y-0.01_z-0.00_align_diff.tif",
r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Aligned DoDs\DoD 070923 SfM\LPM 070923 SfM Veg Masked_aligned_to_LPM 081222_nuth_x+0.00_y+0.01_z+0.00_align_diff.tif",
r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Aligned DoDs\DoD 070923 SfM\MM 070923 SfM Veg Masked_aligned_to_MM 081222_nuth_x-0.03_y-0.06_z+0.02_align_diff.tif",
r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Aligned DoDs\DoD 070923 SfM\MPM 070923 SfM Masked Veg_aligned_to_MPM 100622_nuth_x+0.27_y-0.50_z+0.08_align_diff.tif",
r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Aligned DoDs\DoD 070923 SfM\UM1_070923 - 090822_5cm_veg_masked_QGIS.tif",
r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Aligned DoDs\DoD 070923 SfM\UM2 070923 SfM Masked Veg_aligned_to_UM2 071922_nuth_x-0.02_y+0.00_z+0.00_align_diff.tif",
]
    
    error_rasters = [
        r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\SfM Propagated\LM2_2023_pt_prec_070923_error_prop_ndv.tif",
r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\SfM Propagated\LPM_Intersection_PA3_RMSE_018_pt_prec_070923_error_prop_ndv.tif",
r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\SfM Propagated\MM_all_102023_align60k_intersection_one_checked_pt_prec_070923_error_prop_ndv.tif",
r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\SfM Propagated\MPM_2023_090122_REMOVED_pt_prec_070923_error_prop_ndv.tif",
r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\SfM Propagated\UM1_2023_pt_prec_070923_error_prop_ndv.tif",
r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\SfM Propagated\UM2_2023_pt_prec_070923_error_prop_ndv.tif",
    ]


    for raster_file, error_raster in zip(raster_files, error_rasters):
        raster_name = os.path.basename(raster_file).split(".")[0]
        error_name = os.path.basename(error_raster).split(".")[0]
        print(f"Filtering {raster_name}")
        print(f"Using error raster {error_name}")
        output_dir = os.path.join(os.path.dirname(error_raster), "DoD Error Thresholded")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, os.path.basename(raster_file).split(".")[0][0:40] + "_error_thresholded.tif")
        filter_raster_by_threshold(
            threshold_raster_path=error_raster,
            raster_to_filter_path=raster_file,
            output_path=output_file,
            condition_func=lambda error_data, raster_data: np.abs(raster_data) >= error_data 
        )
        
    
    # clipping_raster = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Terrain Feature Rasters\Slope.tif"
    
    # for raster in raster_files:
    #     outfile = raster.split(".")[0] + "_UTMZone13N.tif"
    #     change_raster_crs(raster, outfile, crs)
    #     clipped_file = raster.split(".")[0] + "_clipped.tif"
    #     clip_raster_by_another_raster_extent(outfile, clipping_raster, clipped_file)
    #     os.remove(outfile)
if __name__ == "__main__":
    main()  