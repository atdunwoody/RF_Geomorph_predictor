import xarray as xr
from utils import decompress_grib_gz
import geopandas as gpd
import rioxarray
import os
import multiprocessing as mp
from tqdm import tqdm
import gzip

input_folders = [
    r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2020",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2021",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2022",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag USA\2023",
]
output_folders = [
    r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped USA\2020",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped\2021",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped\2022",
    # r"Y:\ATD\GIS\MRMS_Data\MRMS PrecipFlag Clipped\2023"
]

MRMS_prefix = "PrecipFlag_00.00_"

shapefile = r"Y:\ATD\GIS\MRMS_Data\ETF_Bennett_Boundaries.gpkg"
gdf = gpd.read_file(shapefile)
total_bounds = gdf.total_bounds

save_as_netcdf = True  # Set to True to save all data in a single NetCDF file, False for GeoTIFF

def modify_MRMS_crs(data, crs="epsg:4326"):
    """
    Modify the CRS of an xarray DataArray containing MRMS data.
    Parameters:
    - data: xarray.DataArray, the MRMS data.
    """
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
    
    if os.path.exists(output_file_path) and not save_as_netcdf:
        return f"File {output_file_path} already exists, skipping."

    try:
        decompress_grib_gz(input_file_path, decompressed_file_path)
    except gzip.BadGzipFile:
        return f"File {input_file_path} is not a valid GZIP file, skipping."

    data = xr.open_dataset(decompressed_file_path, engine='cfgrib', chunks={'time': 10})
    data = modify_MRMS_crs(data)
    clipped_data = clip_data_to_geopackage(data)

    if save_as_netcdf:
        return clipped_data

    clipped_data.rio.to_raster(output_file_path, driver='GTiff')
    del data
    del clipped_data

    return f"Processed {output_file_path}"

if __name__ == "__main__":
    print("Starting script...")
    
    for output_folder_path in output_folders:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

    files_to_process = []
    for input_folder_path, output_folder_path in zip(input_folders, output_folders):
        file_names = [f for f in os.listdir(input_folder_path) if f.endswith('.gz')]
        files_to_process.extend([(input_folder_path, output_folder_path, file_name) for file_name in file_names])

    print(f"Processing {len(files_to_process)} files...")

    if save_as_netcdf:
        all_data = []
        with mp.Pool(mp.cpu_count()) as pool:
            for clipped_data in tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process)):
                if isinstance(clipped_data, xr.DataArray):
                    all_data.append(clipped_data)
        
        if all_data:
            combined_data = xr.concat(all_data, dim='time')
            combined_data.to_netcdf(os.path.join(output_folders[0], "combined_clipped_data.nc"))

    else:
        with mp.Pool(mp.cpu_count()) as pool:
            for result in tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process)):
                print(result)

