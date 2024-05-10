import geopandas as gpd
import pandas as pd
import rasterio
from osgeo import ogr
from datetime import datetime
import os

def open_gpkg(gpkg_path, layer_name):
    """
    Open a layer from a GeoPackage file as a GeoDataFrame.

    Parameters:
    - gpkg_path (str): The file path to the GeoPackage file.
    - layer_name (str): The name of the layer to open from the GeoPackage file.

    Returns:
    - GeoDataFrame: The GeoDataFrame containing the layer from the GeoPackage file.
    """
    return gpd.read_file(gpkg_path, layer=layer_name)

def save_gpkg(gdf, out_path, layer_name = None, overwrite = False):
    """
    Save a GeoDataFrame to a GeoPackage file.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame to save.
    - out_path (str): The file path to save the GeoPackage file.
    - layer_name (str): The name of the layer to save in the GeoPackage file.
    
    """
    if overwrite and layer_name:
        gdf.to_file(out_path, layer=layer_name, driver='GPKG', mode='w')
    elif overwrite:
        gdf.to_file(out_path, driver='GPKG', mode ='w')
    elif layer_name:
        gdf.to_file(out_path, layer=layer_name, driver='GPKG', mode='a')
    else:
        gdf.to_file(out_path, driver='GPKG', mode ='a')
        
def reproject_geopackage_layers(gpkg_path, raster_path, output_gpkg_path):
    """
    Reprojects all layers in a GeoPackage to match the CRS of the given raster.

    Parameters:
    gpkg_path (str): Path to the input GeoPackage.
    raster_path (str): Path to the input raster.
    output_gpkg_path (str): Path to the output GeoPackage.
    """
    # Open the raster to read its CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    # Open the GeoPackage
    driver = ogr.GetDriverByName('GPKG')
    input_gpkg = driver.Open(gpkg_path, 0)  # 0 means read-only
    layer_count = input_gpkg.GetLayerCount()

    # Process each layer in the GeoPackage
    for i in range(layer_count):
        layer = input_gpkg.GetLayerByIndex(i)
        layer_name = layer.GetName()

        # Load the layer with GeoPandas
        gdf = gpd.read_file(gpkg_path, layer=layer_name)

        # Reproject the GeoDataFrame
        gdf = gdf.to_crs(raster_crs.to_string())

        # Save the reprojected layer to the new GeoPackage
        mode = 'a' if i > 0 else 'w'  # Append if not the first layer
        gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG", mode=mode)

    # Clean up
    input_gpkg = None

def add_field_to_gdf(gdf, field_name, value):
    """
    Adds a new field to a GeoDataFrame and populates it with a specified value for all features.
    
    Parameters:
    - gdf: GeoDataFrame to modify.
    - field_name: string, name of the new field to add.
    - value: the value to set for each feature in the new field.
    """
    # Check if the field already exists and raise an error if it does
    if field_name in gdf.columns:
        raise ValueError(f"Field '{field_name}' already exists in the GeoDataFrame.")
    
    # Assign the new field with the specified value
    gdf[field_name] = value
    
    return gdf

def convert_gdf_to_df_and_save_csv(gdf, csv_output_path):
    """
    Converts a GeoDataFrame to a pandas DataFrame and saves it to a CSV file.

    Parameters:
    - gdf: The input GeoDataFrame.
    - csv_output_path: Path where the CSV file will be saved.
    - keep_geometry: Boolean indicating whether to keep the geometry column as a text column.
    """

    # Convert GeoDataFrame to DataFrame by dropping the geometry column if not needed
    df = pd.DataFrame(gdf.drop(columns='geometry')) 

    # Save the DataFrame to CSV
    df.to_csv(csv_output_path, index=False)
    
    print(f"CSV file has been saved to: {csv_output_path}")

def join_shapefiles_to_gpkg(shp_paths, gpkg_path, join_field, columns_to_keep=None):
    """
    Join multiple shapefiles to a GeoPackage based on a common join field, creating new fields if they do not exist,
    and save the updated GeoDataFrame back to the GeoPackage. Optionally keep only specific columns from shapefiles.

    Args:
    shp_paths (list): A list of paths to the shapefiles to be joined.
    gpkg_path (str): Path to the GeoPackage to join onto.
    join_field (str): The field name to join on.
    columns_to_keep (list, optional): List of column names to keep from the shapefiles.

    Returns:
    gpd.GeoDataFrame: The GeoDataFrame after all joins have been completed and saved.
    """
    # Load the GeoPackage
    gdf_main = gpd.read_file(gpkg_path)

    # Iterate through each shapefile path
    for shp_path in shp_paths:
        # Load the shapefile as a GeoDataFrame
        gdf_shp = gpd.read_file(shp_path)
        
        # Filter columns to keep if specified
        if columns_to_keep is not None:
            gdf_shp = gdf_shp[[join_field] + [col for col in columns_to_keep if col in gdf_shp.columns]]

        # Perform the join with suffixes to handle overlapping column names
        gdf_main = gdf_main.merge(gdf_shp, on=join_field, how='left', suffixes=('', '_new'))

        # For each column in the new shapefile, check if it needs to be renamed or created
        for column in gdf_shp.columns:
            if column not in gdf_main.columns:
                # If the column from the shapefile isn't in the main GDF, rename the suffixed column
                gdf_main.rename(columns={f'{column}_new': column}, inplace=True)
            elif '_new' in column:
                # Resolve any potential conflicts not handled by the suffix
                gdf_main[column] = gdf_main.apply(
                    lambda row: row[column] if pd.notna(row[column]) else row[f'{column}_new'], axis=1
                )
                gdf_main.drop(columns=[f'{column}_new'], inplace=True)

    # Save the updated GeoDataFrame back to the GeoPackage
    gdf_main.to_file(gpkg_path, driver='GPKG')

    return gdf_main

def join_df_list_on_field(df_list):
    """
    Join a list of DataFrames on a common field.

    Args:
    df_list (list): List of DataFrames to join.

    Returns:
    pd.DataFrame: The DataFrame after all joins have been completed.
    """
    # Start with the first DataFrame
    df_main = df_list[0]

    # Iterate through the remaining DataFrames
    for df in df_list[1:]:
        # Perform the join
        df_main = df_main.merge(df, on='common_field', how='left')

    return df_main

def add_categories(gpkg_list):
    field_list = ['LM2', 'LPM', 'MM', 'MPM', 'UM1', 'UM2']          
              
    for gpkg, watershed in zip(gpkg_list, field_list):
        gdf = gpd.read_file(gpkg)
        gdf = add_field_to_gdf(gdf, 'Category', "Hillslope")
        save_gpkg(gdf, gpkg, overwrite = True)

def concatenate_csv_list(csv_list, output_csv):
    """
    Concatenate a list of CSV files into a single CSV file.

    Args:
    csv_list (list): List of paths to CSV files to concatenate.
    output_csv (str): Path to the output concatenated CSV file.

    Returns:
    None
    """
    # Read each CSV file into a DataFrame and concatenate them
    df_list = [pd.read_csv(csv) for csv in csv_list]
    df_concat = pd.concat(df_list, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    df_concat.to_csv(output_csv, index=False)

    print(f"CSV files have been concatenated and saved to: {output_csv}")

def main():
    gpkg_list = [
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Hillslopes\Stream Clipped Hillslopes Pruned\LM2 Hillslope Stats Clipped.gpkg",
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Hillslopes\Stream Clipped Hillslopes Pruned\LPM Hillslope Stats Clipped.gpkg",
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Hillslopes\Stream Clipped Hillslopes Pruned\MM Hillslope Stats Clipped.gpkg",
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Hillslopes\Stream Clipped Hillslopes Pruned\MPM Hillslope Stats Clipped.gpkg",
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Hillslopes\Stream Clipped Hillslopes Pruned\UM1 Hillslope Stats Clipped.gpkg",
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Hillslopes\Stream Clipped Hillslopes Pruned\UM2 Hillslope Stats Clipped.gpkg",
                    ]

    #Write a function to read csv file list concatenate and save as a single csv file

    for gpkg in gpkg_list:
        gdf = gpd.read_file(gpkg)
        convert_gdf_to_df_and_save_csv(gdf, gpkg.replace('.gpkg', '.csv'))

    csv_list = [gpkg.replace('.gpkg', '.csv') for gpkg in gpkg_list]

    concatenate_csv_list(csv_list, r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Hillslopes\Stream Clipped Hillslopes Pruned\Hillslope_Stats_Combined 050224.csv") 

if __name__ == "__main__":
    main()