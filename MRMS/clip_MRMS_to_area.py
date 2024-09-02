import xarray as xr
import geopandas as gpd
import rioxarray
import os
import multiprocessing as mp
from tqdm import tqdm
import gc

input_folders = [
    r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2020",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2021",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2022",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2023",
]
output_folders = [
    r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped\2020",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped\2021",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped\2022",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped\2023"
]
MRMS_prefix = "PrecipFlag_00.00_"
shapefile = r"Y:\ATD\GIS\MRMS_Data\ETF_Bennett_Boundaries.gpkg"


gdf = gpd.read_file(shapefile)
total_bounds = gdf.total_bounds

def decompress_grib_gz(file_path, output_path):
    """Decompress a .grib.gz file to a .grib file."""
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    return output_path

def modify_MRMS_crs(data, crs="epsg:4326"):
    data['longitude'] = ((data['longitude'] + 180) % 360) - 180
    data = data.sortby('longitude')
    data.rio.write_crs(crs, inplace=True)
    return data

def clip_data_to_geopackage(data):
    clipped = data.rio.clip_box(*total_bounds)
    return clipped

def process_file(file_paths):
    input_folder_path, output_folder_path, file_name = file_paths
    input_file_path = os.path.join(input_folder_path, file_name)
    decompressed_file_path = os.path.join(output_folder_path, file_name.replace('.gz', ''))
    output_file_name = decompressed_file_path.replace(MRMS_prefix, 'Clipped_').replace('.grib2', '.tif')
    output_file_path = os.path.join(output_folder_path, output_file_name)

    if os.path.exists(output_file_path):
        return False

    decompress_grib_gz(input_file_path, decompressed_file_path)

    data = xr.open_dataset(decompressed_file_path, engine='cfgrib', chunks={'time': 1000})
    data = modify_MRMS_crs(data)

    clipped_data = clip_data_to_geopackage(data)
    clipped_data.rio.to_raster(output_file_path, driver='GTiff')

    del data
    del clipped_data
    gc.collect()

    return True

if __name__ == "__main__":
    for output_folder_path in output_folders:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

    files_to_process = []
    for input_folder_path, output_folder_path in zip(input_folders, output_folders):
        file_names = [f for f in os.listdir(input_folder_path) if f.endswith('.gz')]
        files_to_process.extend([(input_folder_path, output_folder_path, file_name) for file_name in file_names])

    with mp.Pool(mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, files_to_process), total=len(files_to_process)):
            pass
