import numpy as np
import rasterio
import rioxarray
from rasterio.enums import Resampling
from dask.diagnostics import ProgressBar
import glob
import pandas as pd
import os


def gross_error_propagation_LIDAR(SfM_raster_path, output_raster_path, LIDAR_error=0.1):
    #LIDAR_error is the RMSEz reported from the LIDAR data report, 0.1 for 2020 ETF LIDAR data
    SfM_raster = rioxarray.open_rasterio(SfM_raster)
    #perform raster math output = 1.96 * sqrt((SfM/1000)^2 + (0.19)^2)
    output_raster = 1.96 * np.sqrt((SfM_raster/1000)**2 + (LIDAR_error)**2)
    # Save the transformed raster
    with ProgressBar():
        output_raster.rio.to_raster(output_raster_path)


def gross_error_propagation(input_raster1_path, input_raster2_path, output_raster_path):

    print(f"Processing {input_raster1_path}\n{input_raster2_path}")
    input_raster1 = rioxarray.open_rasterio(input_raster1_path, chunks=True)
    input_raster2 = rioxarray.open_rasterio(input_raster2_path, chunks=True)
    #match the resolution of the two rasters
    matched_raster = input_raster1.rio.reproject_match(input_raster2)
    ref_raster = input_raster2
    
    #propagate error accoridng to James 2020
    output_raster = 1.96 * np.sqrt((matched_raster/1000)**2 + (ref_raster/1000)**2)
    #set the nodata value to 3.3347636e+35
    output_raster.rio.write_nodata(3.3347636e+35)
    # Save the transformed raster
    with ProgressBar():
        output_raster.rio.to_raster(output_raster_path)
        
   
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

#write a function that opens a raster, sets the nodata value to 3.3347636e+35, and saves the raster
def set_nodata_value(raster_path, output_raster_path = None):
    if output_raster_path is None:
        output_raster_path = os.path.join(os.path.dirname(raster_path), os.path.basename(raster_path).split(".")[0] + "_ndv.tif")
    
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        data[data == src.nodata] = 0
        data[data >  1e+20] = 0
        profile = src.profile
        profile.update(nodata=0)
        #add another nodata value to the profile
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(data, 1)

import rioxarray

def threshold_by_error(
    input_raster_path: str,
    error_raster_path: str,
    output_raster_path: str
) -> None:
    """
    Thresholds the input_raster by an error_raster. The error_raster is first
    reprojected/matched to the input_raster using rioxarray. Pixels in the
    input_raster are kept if input_raster <= error_raster_matched; otherwise, 
    those pixels are set to NaN.

    Parameters
    ----------
    input_raster_path : str
        File path to the input raster to be thresholded.
    error_raster_path : str
        File path to the error raster used for thresholding.
    output_raster_path : str
        File path where the thresholded raster will be saved.
    """

    # Read in the input raster
    input_da = rioxarray.open_rasterio(input_raster_path, masked=True)

    # Read in the error raster
    error_da = rioxarray.open_rasterio(error_raster_path, masked=True)

    # Match the error raster to the input raster’s coordinate reference system (CRS),
    # resolution, and extent
    error_matched = error_da.rio.reproject_match(input_da)

    # Perform the thresholding
    thresholded_da = input_da.where(input_da <= error_matched)

    # Save the thresholded result to a new raster
    thresholded_da.rio.to_raster(output_raster_path)

def threshold_by_error_lists(
    input_raster_paths: list,
    error_raster_paths: list,
    output_raster_folder: str
) -> None:
    """
    Thresholds a list of input rasters by a list of error rasters. The error
    rasters are first reprojected/matched to the input rasters using rioxarray.
    Pixels in the input rasters are kept if input_raster <= error_raster_matched;
    otherwise, those pixels are set to NaN.

    Parameters
    ----------
    input_raster_paths : list
        List of file paths to the input rasters to be thresholded.
    error_raster_paths : list
        List of file paths to the error rasters used for thresholding.
    output_raster_folder : str
        Folder path where the thresholded rasters will be saved.
    """

    for input_raster_path, error_raster_path in zip(input_raster_paths, error_raster_paths):
        # Create the output raster path
        output_raster_path = os.path.join(
            output_raster_folder,
            os.path.basename(input_raster_path).split(".")[0] + "_gross_error_thresh.tif"
        )
        print(f"Processing:\n{error_raster_path}\n{input_raster_path}")
        # Threshold the input raster by the error raster
        threshold_by_error(input_raster_path, error_raster_path, output_raster_path)
        print(f"Output saved to:\n{output_raster_path}")

def main():

    error_raster_list = [
        r"Y:\ATD\GIS\Bennett\DoDs\Error\Gross Change\Krigged SfM Covariance\Propagated Error\ME_error_propagated.tif",
        r"Y:\ATD\GIS\Bennett\DoDs\Error\Gross Change\Krigged SfM Covariance\Propagated Error\MM_error_propagated.tif",
        r"Y:\ATD\GIS\Bennett\DoDs\Error\Gross Change\Krigged SfM Covariance\Propagated Error\MW_error_propagated.tif",
        r"Y:\ATD\GIS\Bennett\DoDs\Error\Gross Change\Krigged SfM Covariance\Propagated Error\UE_error_propagated.tif",
        r"Y:\ATD\GIS\Bennett\DoDs\Error\Gross Change\Krigged SfM Covariance\Propagated Error\UM_error_propagated.tif",
        r"Y:\ATD\GIS\Bennett\DoDs\Error\Gross Change\Krigged SfM Covariance\Propagated Error\UW_error_propagated.tif",
        ]
    
    veg_mask_raster_list = [
        r"Y:\ATD\GIS\Bennett\DoDs\Masked DoDs\SfM 2023-2022\ME 062023 - 060222 DoD 0,05m_veg_masked.tif",
        r"Y:\ATD\GIS\Bennett\DoDs\Masked DoDs\SfM 2023-2022\MM 062023 - 052022 DoD 0,05m_veg_masked.tif",
        r"Y:\ATD\GIS\Bennett\DoDs\Masked DoDs\SfM 2023-2022\MW 062023 - 052022 DoD 0,05m ndv_veg_masked.tif",
        r"Y:\ATD\GIS\Bennett\DoDs\Masked DoDs\SfM 2023-2022\UE 062023-062022 DoD 0,05m_veg_masked.tif",
        r"Y:\ATD\GIS\Bennett\DoDs\Masked DoDs\SfM 2023-2022\UM 062023-062022 DoD 0,05m_veg_masked.tif",
        r"Y:\ATD\GIS\Bennett\DoDs\Masked DoDs\SfM 2023-2022\UW 062023-062022 DoD 0,05m_veg_masked.tif",
        ]
    
    output_folder= r"Y:\ATD\GIS\Bennett\DoDs\Masked DoDs Gross Error Thresh\SfM 2023-2022"
    
    threshold_by_error_lists(veg_mask_raster_list, error_raster_list, output_folder)
    

if __name__ == "__main__":
    main()