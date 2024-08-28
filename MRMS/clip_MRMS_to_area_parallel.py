import xarray as xr
from utils import decompress_grib_gz
import geopandas as gpd
import rioxarray
import os
import multiprocessing as mp
from tqdm import tqdm

input_folders = [
            r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2021",
            r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2022",
            r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2023",
]
output_folders = [
    r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped\2021",
    r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped\2022",
    r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped\2023"
]

MRMS_prefix = "PrecipFlag_00.00_"

shapefile = r"Y:\ATD\GIS\MRMS_Data\ETF_Bennett_Boundaries.gpkg"
gdf = gpd.read_file(shapefile)
total_bounds = gdf.total_bounds

def modify_MRMS_crs(data, crs="epsg:4326"):
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

# Define the clipping function outside the loop
def clip_data_to_geopackage(data):
    clipped = data.rio.clip_box(*total_bounds)
    return clipped

# Define the processing function to be parallelized
def process_file(file_paths):
    input_folder_path, output_folder_path, file_name = file_paths
    input_file_path = os.path.join(input_folder_path, file_name)
    decompressed_file_path = os.path.join(output_folder_path, file_name.replace('.gz', ''))
    output_file_name = decompressed_file_path.replace(MRMS_prefix, 'Clipped_').replace('.grib2', '.tif')
    output_file_path = os.path.join(output_folder_path, output_file_name)
    if os.path.exists(output_file_path):
        return f"File {output_file_path} already exists, skipping."

    # Decompress the file
    decompress_grib_gz(input_file_path, decompressed_file_path)

    # Load the GRIB2 file using xarray with cfgrib engine
    data = xr.open_dataset(decompressed_file_path, engine='cfgrib', chunks={'time': 10})
    #data = modify_MRMS_crs(data)

    # Clip data to the geopackage bounds
    clipped_data = clip_data_to_geopackage(data)

    # Write clipped data to TIFF
    clipped_data.rio.to_raster(output_file_path, driver='GTiff')    

    # Clean up memory
    del data
    del clipped_data

    return f"Processed {output_file_path}"

if __name__ == "__main__":
    # Create output folders if they don't exist
    for output_folder_path in output_folders:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

    # Prepare list of (input_folder_path, output_folder_path, file_name) tuples
    files_to_process = []
    for input_folder_path, output_folder_path in zip(input_folders, output_folders):
        file_names = [f for f in os.listdir(input_folder_path) if f.endswith('.gz')]
        files_to_process.extend([(input_folder_path, output_folder_path, file_name) for file_name in file_names])

    print(f"Processing {len(files_to_process)} files...")
    # Use multiprocessing to process files in parallel with a progress bar
    with mp.Pool(mp.cpu_count()) as pool:
        for result in tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process)):
            print(result)
