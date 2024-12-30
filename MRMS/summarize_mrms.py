#!/usr/bin/env python3

import os
import glob
from datetime import datetime
import rasterio
import numpy as np
from tqdm import tqdm

def find_raster_files(input_dir, start_dt, end_dt):
    """
    Find all raster files within the input directory that fall within the specified date range.

    Parameters:
        input_dir (str): Directory containing MRMS clipped TIFF files.
            NOTE: Assumes the files are named in the format 'Clipped_YYYYMMDD-HHMMSS.tif'. 
        start_dt (datetime): Start datetime.
        end_dt (datetime): End datetime.

    Returns:
        list: Sorted list of file paths that match the date criteria.
    """
    pattern = os.path.join(input_dir, 'Clipped_*.tif')
    all_files = glob.glob(pattern)
    selected_files = []
    for file in all_files:
        basename = os.path.basename(file)
        # Extract datetime part: Clipped_YYYYMMDD-HHMMSS.tif
        try:
            dt_str = basename.split('_')[1].split('.tif')[0]
            file_dt = datetime.strptime(dt_str, '%Y%m%d-%H%M%S')
            if start_dt <= file_dt <= end_dt:
                selected_files.append(file)
        except (IndexError, ValueError):
            print(f"Skipping file with unexpected name format: {basename}")
    selected_files.sort()
    return selected_files

def aggregate_rasters(file_list, method):
    """
    Summarize raster data using the specified method.

    Parameters:
        file_list (list): List of raster file paths to process.
        method (str): Summarization method ('max' or 'sum').

    Returns:
        tuple: Summarized array and metadata.
    """
    if not file_list:
        raise ValueError("No raster files found for the specified date range.")

    # Initialize summary array
    with rasterio.open(file_list[0]) as src:
        meta = src.meta.copy()
        summary_array = None

    for idx, file in enumerate(tqdm(file_list, desc='Processing rasters')):
        with rasterio.open(file) as src:
            data = src.read(1)  # Assuming single-band rasters
            if summary_array is None:
                summary_array = data.astype('float32')  # Initialize with first raster
            else:
                if method == 'max':
                    summary_array = np.maximum(summary_array, data)
                elif method == 'sum':
                    summary_array += data
                else:
                    raise ValueError("Unsupported method. Choose 'max' or 'sum'.")

    return summary_array, meta

def summarize_mrms(input_directory, output_folder, start_datetime_str, end_datetime_str, summarization_method):
    # Define the output raster file path
    start_datetime_str_label = start_datetime_str.split('-')[0]
    end_datetime_str_label = end_datetime_str.split('-')[0]
    if summarization_method == 'max':
        filename = f'MRMS_MI60_{start_datetime_str_label}-{end_datetime_str_label}.tif'
    else:
        filename = f'MRMS_accum_{start_datetime_str_label}-{end_datetime_str_label}.tif'
    output_raster_path = os.path.join(output_folder, filename)  

    # Define the input directory containing the clipped TIFF files


    # === End of User-Defined Parameters ===

    # Validate and parse datetimes
    try:
        start_dt = datetime.strptime(start_datetime_str, '%Y%m%d-%H%M%S')
    except ValueError:
        raise ValueError(f"Invalid start datetime format: {start_datetime_str}. Expected YYYYMMDD-HHMMSS.")

    try:
        end_dt = datetime.strptime(end_datetime_str, '%Y%m%d-%H%M%S')
    except ValueError:
        raise ValueError(f"Invalid end datetime format: {end_datetime_str}. Expected YYYYMMDD-HHMMSS.")

    if start_dt > end_dt:
        raise ValueError("Start datetime must be earlier than or equal to end datetime.")

    # Find relevant raster files
    raster_files = find_raster_files(input_directory, start_dt, end_dt)
    if not raster_files:
        print("No raster files found in the specified date range.")
        return

    print(f"Found {len(raster_files)} raster files to process.")
    print(f"Summarizing data from {start_dt} to {end_dt} using method: {summarization_method}")
    print(f"Output raster will be saved to: {output_raster_path}")
    # Summarize rasters
    summary_array, meta = aggregate_rasters(raster_files, summarization_method)

    # Update metadata for output
    meta.update(dtype=rasterio.float32, count=1)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_raster_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Write output raster
    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(summary_array, 1)

    print(f"Summarized raster saved to {output_raster_path}")

def main():
    """
    Main function to summarize MRMS raster files.
    """
    # === User-Defined Parameters ===

    # Define the start and end datetime in 'YYYYMMDD-HHMMSS' format
    start_datetime_str = '20220812-000000'  # Example: August 12, 2022, 00:00:00
    end_datetime_str = '20230709-140000'    # Example: July 9, 2023, 14:00:00

    # Choose the summarization method: 'max' or 'sum'
    summarization_method = 'sum'  # Options: 'max', 'sum'
    output_folder = r'Y:\ATD\GIS\MRMS_Data\Summary Data\20230709-20220812'
    input_directory = r'Y:\ATD\GIS\MRMS_Data\MRMS Data Clipped\2022'
    
    summarize_mrms(input_directory, output_folder, start_datetime_str, end_datetime_str, summarization_method)

if __name__ == "__main__":
    main()
