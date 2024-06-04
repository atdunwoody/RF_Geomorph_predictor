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

def main():

#     raster_list = [
#         r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\LM2_2023_pt_prec_070923.tif",
# # r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\LPM_Intersection_PA3_RMSE_018_pt_prec_070923.tif",
# # r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\MM_all_102023_align60k_intersection_one_checked_pt_prec_070923.tif",
# # r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\MPM_2023_090122_REMOVED_pt_prec_070923.tif",
# # r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\UM1_2023_pt_prec_070923.tif",
# # r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\UM2_2023_pt_prec_070923.tif",   
#     ]
    
#     orig_raster_list = [
#        r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\LM2_2023_pt_prec_081222.tif",
# # r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\LPM_pt_prec_081222.tif",
# # r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\MM_pt_prec_090122.tif",
# # r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\MPM_pt_prec_090122.tif",
# # r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\UM1_2023_pt_prec_071822.tif",
# # r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Error\Gross Change\Krigged SfM Covariance\Point precision Metashape\Error krig at native res\UM2_pt_prec_071122.tif",
        
# #     ]
    raster_dir = r"Y:\ATD\Drone Data Processing\Sediment Budgets\ETF\Unmasked DoDs\Aligned LIDAR\1m DoDs\Hillslopes"
    raster_list = glob.glob(raster_dir + "/*.tif")
    csv_output_dir = os.path.join(raster_dir, "DoD Points")
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)
    
    #raster_list = []
    #for raster, orig_raster in zip(raster_list, orig_raster_list):
    for raster in raster_list:
        csv_output_path = csv_output_dir + "/" + raster.split("\\")[-1].split(".")[0] + ".csv"
        print(f"Processing {raster}")
        raster_pixels_to_points(raster, csv_output_path)
        
        #output_raster_name = raster.split("\\")[-1].split(".")[0] + "_error_prop.tif"
        #output_raster_path = os.path.join(output_folder, output_raster_name)
        #gross_error_propagation(raster, orig_raster, output_raster_path)
    

if __name__ == "__main__":
    main()