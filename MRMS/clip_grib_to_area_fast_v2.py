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
from concurrent.futures import ProcessPoolExecutor

# Set up logging
logging.basicConfig(
    filename=r'MRMS\process_grib.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
    """Check if a file is a valid gzipped file by reading its magic number."""
    with open(file_path, 'rb') as file:
        magic_number = file.read(2)
    return binascii.hexlify(magic_number) == b'1f8b'

def modify_MRMS_crs(data, crs="epsg:4326"):
    """Adjust the CRS of the MRMS data and normalize longitude values."""
    data['longitude'] = ((data['longitude'] + 180) % 360) - 180
    data = data.sortby('longitude')
    data.rio.write_crs(crs, inplace=True)
    return data

def clip_data_to_geopackage(data):
    """Clip the MRMS data to the bounding box of the area of interest."""
    return data.rio.clip_box(*total_bounds)

def process_file(file_paths):
    """Process a GRIB2 file compressed as .gz."""
    input_folder_path, file_name = file_paths
    input_file_path = os.path.join(input_folder_path, file_name)

    if not is_gzipped(input_file_path):
        logging.warning(f"Skipping invalid gzipped file: {file_name}")
        return None

    try:
        with gzip.open(input_file_path, 'rb') as f_in, tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as temp_file:
            temp_file.write(f_in.read())
            temp_file_path = temp_file.name

        data = xr.open_dataset(temp_file_path, engine='cfgrib', chunks={'time': 50000})
        data = modify_MRMS_crs(data)
        clipped_data = clip_data_to_geopackage(data)

        os.remove(temp_file_path)
        return clipped_data

    except Exception as e:
        logging.error(f"Error processing file {file_name}: {e}", exc_info=True)
        return None

def save_data_in_chunks(all_data, output_path, year, chunk_size=50000):
    """Save accumulated data to a NetCDF file in chunks."""
    combined_data = xr.concat(all_data, dim='time')
    combined_data.to_netcdf(output_path,
                            mode='a', engine='netcdf4',
                            encoding={'time': {'chunksizes': (chunk_size,)},
                                      'zlib': True})
    all_data.clear()

if __name__ == "__main__":
    logging.info("Starting the processing of GRIB2 files...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_folder_path in input_folders:
        year = os.path.basename(input_folder_path)
        file_names = [f for f in os.listdir(input_folder_path) if f.endswith('.gz')]
        files_to_process = [(input_folder_path, file_name) for file_name in file_names]

        logging.info(f"Processing {len(files_to_process)} files for year {year}...")

        all_data = []
        output_path = os.path.join(output_folder, f"combined_clipped_data_{year}.nc")

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for clipped_data in tqdm(executor.map(process_file, files_to_process), total=len(files_to_process)):
                if isinstance(clipped_data, xr.DataArray):
                    all_data.append(clipped_data)
                    if len(all_data) >= 1000:
                        save_data_in_chunks(all_data, output_path, year)

        if all_data:
            save_data_in_chunks(all_data, output_path, year)

    logging.info("Processing completed.")
