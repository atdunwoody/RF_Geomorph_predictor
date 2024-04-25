import geopandas as gpd

def save_gpkg(gdf, layer_name, out_path, overwrite = False):
    """
    Save a GeoDataFrame to a GeoPackage file.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame to save.
    - out_path (str): The file path to save the GeoPackage file.
    - layer_name (str): The name of the layer to save in the GeoPackage file.
    
    """
    if overwrite:
        gdf.to_file(out_path, layer=layer_name, driver='GPKG', mode='w')
    else:
        gdf.to_file(out_path, layer=layer_name, driver='GPKG')
