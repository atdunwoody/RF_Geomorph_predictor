import numpy as np
import rasterio
import rioxarray
from rasterio.enums import Resampling
from dask.diagnostics import ProgressBar

def apply_math_to_single_raster(input_raster_path, output_raster_path, math_func, no_data_value = None):
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

def gross_error_propagation_LIDAR(raster_list):
    for raster in raster_list:
        outfile = raster.split(".")[0] + "_LIDAR_error_prop.tif"
        #change_raster_crs(raster, outfile, crs)
        #0.100 is LIDAR RMSEz reported for 2020 LIDAR flights
        ru.apply_math_to_raster(raster, outfile, lambda x: 1.96 * np.sqrt(0.100**2 + (x/1000)**2))

def apply_math_to_raster(input_raster1_path, input_raster2_path, output_raster_path, math_func, no_data_value=None):
    """
    Applies a mathematical operation to two rasters based on a user-defined lambda function.

    Parameters:
    - input_raster1_path (str): Path to the first input raster file.
    - input_raster2_path (str): Path to the second input raster file.
    - output_raster_path (str): Path where the transformed raster will be saved.
    - math_func (callable): A function that takes two numpy arrays (input raster data) and returns a numpy array after applying the mathematical transformation.

    Returns:
    - None: The function saves the transformed raster to the specified output path and does not return any value.
    """
    # Open the first input raster
    with rasterio.open(input_raster1_path) as input_rast1:
        data1 = input_rast1.read(1)  # Read the first band
        profile = input_rast1.profile  # Get the metadata to write back the output
    
    # Open the second input raster
    with rasterio.open(input_raster2_path) as input_rast2:
        data2 = input_rast2.read(1)  # Read the first band
    
    # Filter out no data values
    if no_data_value is None:
        data1 = np.where(data1 == input_rast1.nodata, np.nan, data1)
        data2 = np.where(data2 == input_rast2.nodata, np.nan, data2)
    else:
        data1 = np.where(data1 == no_data_value, np.nan, data1)
        data2 = np.where(data2 == no_data_value, np.nan, data2)
    
    # Apply the mathematical operation using the provided lambda function
    transformed_data = math_func(data1, data2)

    # Save the transformed raster
    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(transformed_data, 1)

def gross_error_propagation(raster_list):
    for input_raster1_path, input_raster2_path in raster_list:
        output_raster_path = input_raster1_path.split(".")[0] + "_error_prop.tif"
        apply_math_to_raster(
            input_raster1_path, 
            input_raster2_path, 
            output_raster_path, 
            lambda x, y: 1.96 * np.sqrt((x/1000)**2 + (y/1000)**2)
        )
        
def match_rasters(source_raster_fn, ref_raster_fn, output_fn=None, match_res=True, method=Resampling.bilinear):
    raster1 = rioxarray.open_rasterio(source_raster_fn, chunks=True)
    raster2 = rioxarray.open_rasterio(ref_raster_fn, chunks=True)

    # Resample raster1 to match the resolution of raster2 (finer resolution)
    resampled_raster = resample_raster_to_match(raster1, raster2, match_res = match_res, method=method)

    if output_fn is not None:
        # Save the resampled raster to a new file
        with ProgressBar():
            resampled_raster.rio.to_raster(output_fn)
    return resampled_raster

def resample_raster_to_match(raster_to_resample, reference_raster, match_res=True, method=Resampling.bilinear):
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

def resample_raster_to_resolution(raster, target_resolution, method= Resampling.bilinear):
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

import rasterio
from rasterio.enums import Resampling
import pandas as pd
import numpy as np

def raster_pixels_to_points(raster_path, output_csv_path, sample_factor=1):
    """
    Convert raster pixels to points, excluding null, NaN, and no data values,
    and save them to a CSV file.

    Parameters:
    - raster_path: str, path to the raster file.
    - output_csv_path: str, path to the output CSV file.
    - sample_factor: int, factor by which to downsample the raster for processing.
    """
    with rasterio.open(raster_path) as src:
        # Resample raster if necessary
        if sample_factor > 1:
            data = src.read(
                out_shape=(
                    src.count,
                    int(src.height / sample_factor),
                    int(src.width / sample_factor)
                ),
                resampling=Resampling.bilinear
            )
        else:
            data = src.read(1)

        # Get coordinates for each pixel
        rows, cols = np.indices(data.shape)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset='center')

        # Flatten arrays and create dataframe
        df = pd.DataFrame({
            'X': np.array(xs).flatten(),
            'Y': np.array(ys).flatten(),
            'Value': data.flatten()
        })

        # Remove no data values
        no_data_value = src.nodatavals[0]
        if no_data_value is not None:
            df = df[df['Value'] != no_data_value]

        # Further clean data to exclude NaN or None values
        df = df.dropna(subset=['Value'])

    # Save to CSV, only non-null, non-NaN, and valid data values
    df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")



def main():
    raster_path = r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Aligned DoDs\DoD 070923 SfM\LM2 070923 SfM Veg Masked\DoD DoD LM2_070923_5cm_veg_masked_QGIS_LM2_081222_5cm_veg_masked_QGIS_nuth_x-0 - LM2_081222_5cm_veg_masked_QGIS.tif"
    csv_output_path = r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Net Change\DoD Points\LM2 070923 - 081222 DoD veg mask.csv"
    raster_pixels_to_points(raster_path, csv_output_path)
if __name__ == "__main__":
    main()