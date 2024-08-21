import os
from pathlib import Path
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import rioxarray


def mask_tif_by_geometries(raster_paths, shapefile_path, output_folder, delete_intermediates = True, verbose=False):
    """
    Mask a list of rasters by all polygons in a shapefile. 
    Only the portions of the raster within the bounds of the polygons will be retained.
    
    Parameters:
    raster_paths (list of str): List of file paths of rasters to be masked.
    shapefile_path (str): File path to the masking shapefile.
    output_folder (str): Folder path to save the masked rasters.

    Returns:
    raster_outputs (list of str): List of file paths to the masked raster files.
    """
    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Read the shapefile
    if verbose:
        print(f"Reading shapefile from {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    masked_rasters = []
    
    for index, geometry in enumerate(gdf.geometry):
        shape_geometry = [geometry.__geo_interface__]  # Convert geometry to GeoJSON-like dict
        
        output_subfolder = Path(output_folder) / f"Masked_{index}"
        output_subfolder.mkdir(parents=True, exist_ok=True)

        for raster_path in raster_paths:
            if verbose:
                print(f"Masking raster {raster_path} using geometry index {index}")
            with rasterio.open(raster_path) as src:
                out_image, out_transform = mask(src, shape_geometry, crop=True)
                out_meta = src.meta.copy()

                # Update metadata for the masked raster
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                })

                # Construct the output raster file path
                raster_filename = os.path.basename(raster_path)
                masked_raster_filename = f"{os.path.splitext(raster_filename)[0]}_masked_{index}.tif"
                masked_raster_path = output_subfolder / masked_raster_filename

                #Clean up the out image
                out_image[out_image > 1e+20] = 0
                out_image[out_image < -1e+20] = 0
                
                
                # Save the masked raster
                with rasterio.open(masked_raster_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                if verbose:
                    print(f"Masked raster saved to {masked_raster_path}")
                masked_rasters.append(str(masked_raster_path))
    if delete_intermediates:
        for raster_path in masked_rasters:
            os.remove(raster_path)
    return masked_rasters

def erase_tif_by_geometries(raster_paths, shapefile_path, output_folder, verbose=False):
    """
    Erase parts of a list of rasters using all polygons in a shapefile. 
    Only the portions of the raster outside the bounds of the polygons will be retained.
    
    Parameters:
    raster_paths (list of str): List of file paths of rasters to be erased.
    shapefile_path (str): File path to the shapefile used for erasing.
    output_folder (str): Folder path to save the erased rasters.

    Returns:
    raster_outputs (list of str): List of file paths to the erased raster files.
    """
    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Read the shapefile
    if verbose:
        print(f"Reading shapefile from {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)

    erased_rasters = []
    
    for index, geometry in enumerate(gdf.geometry):
        shape_geometry = [geometry.__geo_interface__]  # Convert geometry to GeoJSON-like dict
        
        output_subfolder = Path(output_folder) / f"Erased_{index}"
        output_subfolder.mkdir(parents=True, exist_ok=True)

        for raster_path in raster_paths:
            if verbose:
                print(f"Erasing raster {raster_path} using geometry index {index}")
            with rasterio.open(raster_path) as src:
                # Invert the mask to erase using the geometry
                out_image, out_transform = mask(src, shape_geometry, crop=False, invert=True)
                out_meta = src.meta.copy()

                if src.nodata is None:
                    nodata = 0  # Set nodata value to 0 if it is None
                else:
                    nodata = src.nodata
                #clean up the out image
                out_image[out_image > 1e+20] = 0
                out_image[out_image < -1e+20] = 0
                
                # Update metadata for the erased raster
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": nodata  # Use original nodata value
                })

                # Construct the output raster file path
                raster_filename = os.path.basename(raster_path)
                erased_raster_filename = f"{os.path.splitext(raster_filename)[0]}_erased_{index}.tif"
                erased_raster_path = output_subfolder / erased_raster_filename
                
                # Save the erased raster
                with rasterio.open(erased_raster_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                if verbose:
                    print(f"Erased raster saved to {erased_raster_path}")
                erased_rasters.append(str(erased_raster_path))

    return erased_rasters


def main():
    
    
    raster_paths = [
       r"Y:\ATD\Drone Data Processing\Metashape_Processing\BlueLake_JoeWright\240723 Blue Lake\Exports\072024-matched.tif",
        r"Y:\ATD\Drone Data Processing\Metashape_Processing\BlueLake_JoeWright\240723 Blue Lake\Exports\082021-matched.tif"
    ]
    
    shp_paths = [
        r"Y:\ATD\Drone Data Processing\Metashape_Processing\BlueLake_JoeWright\240723 Blue Lake\Exports\stable_ground_single.gpkg",
        r"Y:\ATD\Drone Data Processing\Metashape_Processing\BlueLake_JoeWright\240723 Blue Lake\Exports\stable_ground_single.gpkg"
    ]
    output_directory = os.path.join(os.path.dirname(raster_paths[0]), "Masked")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for raster_path, shp_path in zip(raster_paths, shp_paths):
        mask_tif_by_geometries([raster_path], shp_path, output_directory, verbose=True)
    
    # output_directory = os.path.join(os.path.dirname(raster_paths[0]), "Channels")
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    # for raster_path, shp_path in zip(raster_paths, shp_paths):
    #     mask_tif_by_geometries([raster_path], shp_path, output_directory, verbose=True)

if __name__ == "__main__":
    main()