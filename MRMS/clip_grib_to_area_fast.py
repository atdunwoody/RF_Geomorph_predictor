import xarray as xr
import geopandas as gpd
import rioxarray
import os
import multiprocessing as mp
from tqdm import tqdm
import gzip
import tempfile
import binascii
import logging

# Set up logging
logging.basicConfig(
    filename=r'MRMS\process_grib.log',  # Log file name
    level=logging.INFO,  # Log level: INFO means general events
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)

# Input folders containing the GRIB2 files compressed as .gz, organized by year
input_folders = [
    r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2020",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2021",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2022",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2023",
]

# Output folder where the final NetCDF files will be saved
output_folder = r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped"

# Shapefile defining the area of interest to clip the data
shapefile = r"Y:\ATD\GIS\MRMS_Data\ETF_Bennett_Boundaries.gpkg"
gdf = gpd.read_file(shapefile)
total_bounds = gdf.total_bounds  # Get the bounding box coordinates

def is_gzipped(file_path):
    """
    Check if a file is a valid gzipped file by reading its magic number.
    """
    with open(file_path, 'rb') as file:
        magic_number = file.read(2)
    return binascii.hexlify(magic_number) == b'1f8b'

def modify_MRMS_crs(data, crs="epsg:4326"):
    """
    Adjust the Coordinate Reference System (CRS) of the MRMS data.
    Ensures that the longitude values are within the range -180 to 180 degrees.
    Args:
    - data: xarray.DataArray containing the MRMS data.
    - crs: String representing the target CRS (default is "epsg:4326").
    Returns:
    - Modified xarray.DataArray with updated CRS.
    """
    data['longitude'] = ((data['longitude'] + 180) % 360) - 180  # Normalize longitudes
    data = data.sortby('longitude')  # Sort by longitude for proper alignment
    data.rio.write_crs(crs, inplace=True)  # Assign the CRS to the data
    return data

def clip_data_to_geopackage(data):
    """
    Clip the MRMS data to the bounding box of the area of interest.
    Args:
    - data: xarray.DataArray containing the MRMS data.
    Returns:
    - Clipped xarray.DataArray.
    """
    clipped = data.rio.clip_box(*total_bounds)  # Clip the data to the specified bounds
    return clipped

def process_file(file_paths):
    """
    Process an individual GRIB2 file compressed as .gz.
    The file is decompressed and saved to a temporary file, the data is loaded, CRS is modified,
    and the data is clipped to the area of interest.
    Args:
    - file_paths: Tuple containing the input folder path and file name.
    Returns:
    - Clipped xarray.DataArray.
    """
    input_folder_path, file_name = file_paths
    input_file_path = os.path.join(input_folder_path, file_name)

    # Check if the file is a valid gzipped file
    if not is_gzipped(input_file_path):
        logging.warning(f"Skipping invalid gzipped file: {file_name}")
        return None

    try:
        # Decompress the .gz file and write it to a temporary file
        with gzip.open(input_file_path, 'rb') as f_in:
            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as temp_file:
                temp_file.write(f_in.read())
                temp_file_path = temp_file.name

        # Load the decompressed data from the temporary file as an xarray DataArray using the cfgrib engine
        data = xr.open_dataset(temp_file_path, engine='cfgrib', chunks={'time': 100})

        # Modify the CRS to ensure proper alignment
        data = modify_MRMS_crs(data)

        # Clip the data to the defined area of interest
        clipped_data = clip_data_to_geopackage(data)

        # Clean up the temporary file
        os.remove(temp_file_path)

        return clipped_data

    except Exception as e:
        logging.error(f"Error processing file {file_name}: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    logging.info("Starting the processing of GRIB2 files...")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process files for each year separately
    for input_folder_path in input_folders:
        year = os.path.basename(input_folder_path)  # Extract the year from the folder name

        # Prepare the list of files to process
        files_to_process = []
        file_names = [f for f in os.listdir(input_folder_path) if f.endswith('.gz')]
        files_to_process.extend([(input_folder_path, file_name) for file_name in file_names])

        logging.info(f"Processing {len(files_to_process)} files for year {year}...")

        all_data = []  # Initialize a list to hold the processed data for this year
        with mp.Pool(mp.cpu_count()) as pool:
            # Use multiprocessing to process files in parallel
            for clipped_data in tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process)):
                if isinstance(clipped_data, xr.DataArray):
                    all_data.append(clipped_data)  # Append the processed data to the list
                    
                    # Save to NetCDF incrementally every 1000 files to manage memory
                    if len(all_data) >= 1000:  # Adjust the threshold as needed
                        combined_data = xr.concat(all_data, dim='time')  # Concatenate along the time dimension
                        combined_data.to_netcdf(os.path.join(output_folder, f"combined_clipped_data_{year}.nc"),
                                                mode='a', engine='netcdf4',  # Append to the existing NetCDF file
                                                encoding={'time': {'chunksizes': (1000,)},  # Optimize chunking
                                                          'zlib': True})  # Enable compression
                        all_data.clear()  # Clear the list to free up memory

        # Final save for any remaining data that wasn't saved in the loop
        if all_data:
            combined_data = xr.concat(all_data, dim='time')
            combined_data.to_netcdf(os.path.join(output_folder, f"combined_clipped_data_{year}.nc"),
                                    mode='a', engine='netcdf4',
                                    encoding={'time': {'chunksizes': (1000,)},
                                              'zlib': True})

    logging.info("Processing completed.")
