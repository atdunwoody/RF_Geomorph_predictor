import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
import os
from vector_utils import save_gpkg
from shapely.geometry import Point

def add_area_field(gdf, output_shapefile_path=None, id_field='geometry'):

    if type(gdf) == str:
        gdf = gpd.read_file(gdf)
        
    # Check if the id_field exists
    if id_field not in gdf.columns:
        raise ValueError(f"The specified id_field '{id_field}' does not exist in the shapefile.")
    
    # Compute area and perimeter for each sub-shape
    gdf['Area'] = gdf['geometry'].area
    #gdf['Perimeter'] = gdf['geometry'].length
    
    # Optionally, save the updated GeoDataFrame to a new shapefile
    if output_shapefile_path is not None:
        gdf.to_file(output_shapefile_path)
        
    return gdf


def aggregate_elevation(gdf, raster_path = None, output_shp_path = None):
    print("Aggregating Elevation")
    if raster_path is None:
        raster_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LIDAR\Reprojected to UTM Zone 13N\ET_merged_LIDAR_2020_1m_DEM_reproj.tif"
    stats_fields = { 'mean': 'Mean Elevation'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='mean', **stats_fields)
    stats_fields = { 'max': 'Max Elevation'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='max', **stats_fields)
    stats_fields = { 'min': 'Min Elevation'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='min', **stats_fields)
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf

def aggregate_slope(gdf, output_shp_path = None):
    print("Aggregating Slope")
    raster_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Terrain Feature Rasters\Slope.tif"
    stats_fields = {'mean': 'Slope Mean'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='mean',  
                        output_shapefile_path=None, **stats_fields)
    
    stats_fields = { 'std': 'Slope std'}
    
    gdf = aggregate_raster_stats(raster_path, gdf, mode='std', 
                        output_shapefile_path=None, **stats_fields)
    
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf

def aggregate_dNBR(gdf, raster_path = None, output_shp_path = None):
    print("Aggregating dNBR")
    dNBR_values = {"Unburned" : 1,
                   "Low Severity" : 2,
                   "Moderate Severity" : 3,
                   "High Severity" : 4,
                    }
    if raster_path is None:
        raster_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\dNBR\east_troublesome_co4020310623920201014_sbs.tif"
    #get percent of dNBR values that match the values in dNBR_values
    for key, value in dNBR_values.items():
        stats_fields = { 'percent': f'dNBR % {key}'}
        gdf = aggregate_raster_stats(raster_path, gdf, mode='percent', raster_value_to_match= value, **stats_fields)
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf
    
def add_centroid_field(gdf, geopackage_output_path = None):
    """
    This function loads a geopackage containing multipolygon geometries, computes the centroids of each multipolygon,
    and saves the result in a new geopackage with an added field 'Centroid' containing the centroid coordinates.
    
    Parameters:
        geopackage_input_path (str): The file path to the input geopackage.
        geopackage_output_path (str): The file path to the output geopackage where the modified data will be saved.
    """
    # Load the geopackage into a GeoDataFrame
    
    # Calculate the centroid for each geometry in the DataFrame
    gdf['Centroid'] = gdf['geometry'].centroid
    
    # Convert Centroid geometry to text representation for easier reading and use
    gdf['Centroid'] = gdf['Centroid'].apply(lambda x: f"({x.x}, {x.y})")
    
    if geopackage_output_path is not None:
        # Save the modified GeoDataFrame back to a new geopackage
        gdf.to_file(geopackage_output_path, driver='GPKG')
    return gdf

def calculate_polygon_width_and_length(polygon):
    # Get the minimum rotated rectangle for the polygon
    mrr = polygon.minimum_rotated_rectangle
    
    # Extract the points of the rectangle
    coords = list(mrr.exterior.coords)
    
    # Calculate the distances between the points (rectangle sides)
    side_lengths = [Point(coords[i]).distance(Point(coords[i+1])) for i in range(len(coords)-1)]
    
    # Sort lengths to get width and length (width < length)
    side_lengths.sort()
    width = side_lengths[0]
    length = side_lengths[-1]
    # Return width and length
    return width, length

def aggregate_width_over_length(gdf, output_shapefile_path=None):
    print("Aggregating Width over Length")
    # Calculate the ratio of width to length
    try:
        gdf['width_over_length'] = gdf['width'] / gdf['length']
    except KeyError:
        gdf['width'], gdf['length'] = zip(*gdf['geometry'].apply(calculate_polygon_width_and_length))
        gdf['width_over_length'] = gdf['width'] / gdf['length']
    # Optionally, save the modified GeoDataFrame to a new shapefile
    if output_shapefile_path is not None:
        gdf.to_file(output_shapefile_path)
    
    return gdf

def aggregate_raster_stats(raster_path, gdf, mode='mean', threshold=None, threshold_direction='above', 
                           raster_value_to_match=None, output_shapefile_path=None, **stats_fields):
    """Aggregate statistics from a raster based on a GeoDataFrame shape.

    Args:
        raster_path (str): Path to the raster file.
        gdf (GeoDataFrame): GeoDataFrame containing the shapes.
        mode (str): Aggregation mode, one of 'mean','median', 'max', 'min', 'std', 'sum', 'percent'.
        threshold (float, optional): Value to filter the raster data.
        threshold_direction (str): 'above' or 'below', direction for threshold filtering.
        raster_value_to_match: Value(s) that raster cells must match to be included.
        output_shapefile_path (str, optional): Path to save the modified GDF.
        **stats_fields: Field names for storing results in GDF, like mean_field='mean_value'.
    
    Returns:
        GeoDataFrame: The updated GeoDataFrame.
    """
    valid_stat_list = ['mean', 'median', 'max', 'min', 'std', 'sum', 'percent', 'count']
    with rasterio.open(raster_path) as src:
        no_data = src.nodata

        # Prepare the GDF by adding new columns for specified stats
        for stat in valid_stat_list:
            field_name = stats_fields.get(f'{stat}', None)
            if field_name is not None:
                gdf[field_name] = np.nan

        # Process each geometry
        for index, row in gdf.iterrows():
            print(f"Progress: {index+1}/{len(gdf)}", end='\r')
            shapes = gdf.iloc[[index]]
            out_image, out_transform = mask(src, shapes.geometry, crop=True, all_touched=True)
            data = out_image[out_image != no_data]
            data = data[data != 0]

            # Filter data by raster_value_to_match if specified
            if raster_value_to_match is not None:
                if isinstance(raster_value_to_match, (list, tuple)):
                    data_mask = np.isin(data, raster_value_to_match)
                else:
                    data_mask = data == raster_value_to_match
                data_masked = data[data_mask]

            # Apply threshold filtering if specified
            if threshold is not None:
                if threshold_direction == 'above':
                    data = data[data > threshold]
                elif threshold_direction == 'below':
                    data = data[data < threshold]

            # Calculate statistics based on mode
            if data.size > 0:
                if mode == 'mean':
                    result = np.mean(data)
                elif mode == 'median':
                    result = np.median(data)
                elif mode == 'std':
                    result = np.std(data)
                elif mode == 'sum':
                    result = np.sum(data)
                elif mode == 'percent':
                    result = np.count_nonzero(data_masked) / np.count_nonzero(data) * 100
                elif mode == 'count':
                    result = np.count_nonzero(data)
                elif mode == 'max':
                    result = np.max(data)
                elif mode == 'min':
                    result = np.min(data)
            else:
                result = np.nan

            # Update the GDF with the calculated results
            for stat in valid_stat_list:
                field_name = stats_fields.get(f'{stat}', None)
                if field_name is not None and mode == stat:
                    gdf.at[index, field_name] = result

        # Output the modified shapefile if a path is provided
        if output_shapefile_path is not None:
            os.makedirs(os.path.dirname(output_shapefile_path), exist_ok=True)
            gdf.to_file(output_shapefile_path)

        return gdf

def merge_attributes_by_id(geopackage_path, layer, id_tuples, out_path):
    """
    Merges features within a GeoDataFrame based on specified IDs and recalculates attributes as weighted averages.

    Args:
        geopackage_path (str): Path to the geopackage file.
        layer (str): Name of the layer within the geopackage to process.
        id_tuples (list of tuples): List of tuples, each containing IDs of features to merge.
        out_path (str): Output path to save the modified geopackage.

    Returns:
        None: The function does not return a value but saves the modified geopackage to the specified path.
    """
    
    # Load the data from the Geopackage
    gdf = gpd.read_file(geopackage_path, layer = layer)
    
    # Placeholder for the new geometries and updated data
    new_features = []

    for id_group in id_tuples:
        # Filter the geodataframe to only include the relevant IDs
        subset = gdf[gdf['ID'].isin(id_group)]
        
        if subset.empty:
            continue

        # Merge the geometries into a single geometry
        merged_geometry = subset.geometry.unary_union
        
        # Calculate weighted average of the attributes
        # Assuming 'area' is a field in the dataframe, and 'attribute1', 'attribute2'... need to be averaged
        exclude_columns = ['fid', 'ID', 'geometry', 'area']
        attribute_columns = [col for col in subset.columns if col not in exclude_columns]

        weighted_attributes = {}
        total_area = subset['area'].sum()
        for col in attribute_columns:
            weighted_attributes[col] = (subset[col] * subset['area']).sum() / total_area
        
        # Create a new row for the merged feature
        new_feature = subset.iloc[0].copy()
        new_feature['geometry'] = merged_geometry
        for attr, value in weighted_attributes.items():
            new_feature[attr] = value
        
        # Append the new feature to the list of new features
        new_features.append(new_feature)

    # Create a new GeoDataFrame with the new features
    new_gdf = gpd.GeoDataFrame(new_features, crs=gdf.crs)

    # Optionally: merge new_gdf back with gdf to include non-merged features
    remaining_features = gdf[~gdf['ID'].isin([id for group in id_tuples for id in group])]
    final_gdf = pd.concat([remaining_features, new_gdf], ignore_index=True)

    # Save the modified geodataframe back into a Geopackage
    final_gdf.to_file(out_path, layer=layer, driver='GPKG')

def group_features_by_ID(csv_path):
    """
    Groups features by a 'Group' column in a CSV file and collects corresponding 'Feature ID' values into lists.

    Args:
        filepath (str): Path to the CSV file containing feature data with 'Group' and 'Feature ID' columns.

    Returns:
        list of lists: Each sublist contains 'Feature ID' values that share the same group.
    """
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Group the DataFrame by the 'Group' column and collect the 'Feature ID' values into lists
    grouped = df.groupby('Group')['Feature ID'].apply(list)

    # Convert the grouped data into a list of tuples
    grouped_tuples = [(group, features) for group, features in grouped.items()]
    #create a list of just the tuples
    grouped_tuples = [tup[1] for tup in grouped_tuples]
    return grouped_tuples

def aggregate_deposition(gdf, raster_path, output_shp_path = None):
    
    print("Aggregating Deposition Sum")
    stats_fields = { 'sum': 'Deposition Volsum'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='sum', threshold=0, threshold_direction='above'
                                 , output_shapefile_path=None, **stats_fields)
    print("Aggregating Deposition Cell count")
    stats_fields = {'count': 'Deposition Volsum Count'}
    
    gdf = aggregate_raster_stats(raster_path, gdf, mode='count', threshold=0, threshold_direction='above',
                            output_shapefile_path=None, **stats_fields)
    
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    
    return gdf

def aggregate_erosion(gdf, raster_path, output_shp_path = None):
   
    print("Aggregating Erosion Sum")
    stats_fields = { 'sum': 'Erosion Volsum'}
    
    gdf = aggregate_raster_stats(raster_path, gdf, mode='sum', threshold=0, threshold_direction='below', 
                           output_shapefile_path=None, **stats_fields)
    
    print("Aggregating Erosion Cell count")
    stats_fields = { 'count': 'Erosion Volsum Count'}
    
    gdf = aggregate_raster_stats(raster_path, gdf, mode='count', threshold=0, threshold_direction='below', 
                           output_shapefile_path=None, **stats_fields)
    
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    
    return gdf

def aggregate_masked_deposition(gdf, raster_path, output_shp_path = None):
    
    print("Aggregating Deposition Volsum Masked")
    stats_fields = { 'sum': 'Deposition Volsum Masked'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='sum', threshold=0, threshold_direction='above', 
                           output_shapefile_path=None, **stats_fields)
   
    print("Aggregating Deposition Volsum Masked Count")
    stats_fields = {'count': 'Deposition Volsum Masked Count'}
    
    gdf = aggregate_raster_stats(raster_path, gdf, mode='count', threshold=0, threshold_direction='above',
                            output_shapefile_path=None, **stats_fields)
       
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf

def aggregate_masked_erosion(gdf, raster_path, output_shp_path = None):
    print("Aggregating Erosion Volsum Masked")
    stats_fields = { 'sum': 'Erosion Volsum Masked'}
    
    gdf = aggregate_raster_stats(raster_path, gdf, mode='sum', threshold=0, threshold_direction='below',
                            output_shapefile_path=None, **stats_fields)

    print("Aggregating Erosion Volsum Masked Count")
    stats_fields = { 'count': 'Erosion Volsum Masked Count'}
    
    gdf = aggregate_raster_stats(raster_path, gdf, mode='count', threshold=0, threshold_direction='below', 
                           output_shapefile_path=None, **stats_fields)
    
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf

def aggregate_aspect(gdf, output_shp_path = None):
    print("Aggregating Aspect")
    raster_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Terrain Feature Rasters\Aspect.tif"
    stats_fields = { 'mean': 'Aspect Mean'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='mean', 
                        output_shapefile_path=None, **stats_fields)
    
    stats_fields = { 'std': 'Aspect std'}
    
    gdf = aggregate_raster_stats(raster_path, gdf, mode='std', 
                        output_shapefile_path=None, **stats_fields)
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf

def aggregate_bare_earth(gdf, raster_path, output_shp_path = None):
        print("Aggregating Bare Earth Coverage")
        stats_fields = { 'percent': '% Bare Earth'}
        gdf = aggregate_raster_stats(raster_path, gdf, mode='percent', 
                                     raster_value_to_match= [4, 5],  **stats_fields)
        if output_shp_path is not None:
            save_gpkg(gdf, output_shp_path, overwrite=True)
        return gdf
    
def aggregate_MI60(gdf, raster_path = None, output_shp_path = None):
    print("Aggregating Max Intensity 60 min")
    stats_fields = { 'mean': 'Max Int 60 min'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='mean', **stats_fields)
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf

def aggregare_accumulated_precip(gdf, raster_path = None, output_shp_path = None):
    print("Aggregating Accumulated Precipitation")
    stats_fields = { 'mean': 'Accumulated Precipitation'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='mean', **stats_fields)
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf

def aggregate_flow_accumulation(gdf, raster_path = None, output_shp_path = None):
    print("Aggregating Flow Accumulation")
    stats_fields = { 'mean': 'Flow Accumulation'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='mean', **stats_fields)
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf

def aggregate_TRI(gdf, raster_path = None, output_shp_path = None):
    print("Aggregating Terrain Ruggedness Index")
    stats_fields = { 'mean': 'Terrain Ruggedness Index'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='mean', **stats_fields)
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf

def aggregate_curvature(gdf, raster_path = None, output_shp_path = None):
    print("Aggregating curvature")
    stats_fields = { 'mean': 'Curvature'}
    gdf = aggregate_raster_stats(raster_path, gdf, mode='mean', **stats_fields)
    if output_shp_path is not None:
        save_gpkg(gdf, output_shp_path, overwrite=True)
    return gdf

def summarize_raster(gdf):  
    gdf = add_area_field(gdf)
    gdf = aggregate_elevation(gdf)
    gdf = aggregate_slope(gdf)
    gdf = aggregate_dNBR(gdf)
    return gdf
    
    
def main():

    shp_id = 'geometry'
    gpkg_list = [    
        r"Y:\ATD\GIS\East_Troublesome\Watershed_Boundaries\LM2 boundary.gpkg",
        r"Y:\ATD\GIS\East_Troublesome\Watershed_Boundaries\LPM boundary.gpkg",
        r"Y:\ATD\GIS\East_Troublesome\Watershed_Boundaries\MM boundary.gpkg",
        r"Y:\ATD\GIS\East_Troublesome\Watershed_Boundaries\MPM boundary.gpkg",
        r"Y:\ATD\GIS\East_Troublesome\Watershed_Boundaries\UM1 boundary.gpkg",
        r"Y:\ATD\GIS\East_Troublesome\Watershed_Boundaries\UM2 boundary.gpkg",
    ]
    out_gpkg = r"Y:\ATD\GIS\East_Troublesome\Watershed_Boundaries\Watershed_Boundaries_features.gpkg"
 
    for gpkg in gpkg_list:
        gdf = gpd.read_file(gpkg, layer = 1)
        #loop through all layers in the geopackage
        #gdf = summarize_raster(gdf)
        out_gpkg = gpkg.replace('.gpkg', '_features.gpkg')
        #aggregate_bare_earth(gdf, raster, gpkg)
        #gdf_temp = aggregate_MI60(gdf, raster_paths[1])
        #aggregare_accumulated_precip(gdf_temp, raster_paths[0], gpkg)
        # Calculate width and length and store them in new columns
        # gdf['width'], gdf['length'] = zip(*gdf['geometry'].apply(calculate_polygon_width_and_length))
        # gdf = add_centroid_field(gdf)
        # gdf = aggregate_width_over_length(gdf)
        # gdf = aggregate_curvature(gdf, raster_paths[0])
        # gdf = aggregate_TRI(gdf, raster_paths[1])
        # gdf = aggregate_flow_accumulation(gdf, raster_paths[2])
        # gdf = aggregate_erosion_deposition(gdf)
        # gdf = aggregate_masked_erosion(gdf, raster_path)
        # gdf = aggregate_masked_deposition(gdf, raster_path)
        # Save the modified GeoDataFrame to a new shapefile
        gdf.to_file(out_gpkg, driver='GPKG')

if __name__ == '__main__':
    main()