import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import os



def get_raster_stats_and_update_shapefile(raster_path, shapefile_path, **kwargs):
    # Open the raster
    shp_id_field = kwargs.get('shapefile_ID', 'ID') # Name of field in the shapefile where grouping statistics will be calculated
    mean_field_name = kwargs.get('mean_field_name', None)
    std_dev_field_name = kwargs.get('std_dev_field_name', None)
    percent_field_name = kwargs.get('percent_field_name', None)
    raster_val_to_match = kwargs.get('raster_val_to_match', None)
    output_shapefile_path = kwargs.get('output_shapefile_path', os.path.splitext(shapefile_path)[0] + '_updated.shp')
    if not os.path.exists(os.path.dirname(output_shapefile_path)):
        os.makedirs(os.path.dirname(output_shapefile_path))
    progress_counter = 0
    with rasterio.open(raster_path) as src:
        gdf = gpd.read_file(shapefile_path)

        # Add new columns for mean and standard deviation in the GeoDataFrame
        if mean_field_name is not None:
            gdf[mean_field_name] = np.nan
        if std_dev_field_name is not None:
            gdf[std_dev_field_name] = np.nan
        
        # Iterate through each unique ID in the shapefile
        for unique_id in gdf[shp_id_field].unique():
            # Filter the GeoDataFrame to include only the current ID
            shapes = gdf[gdf[shp_id_field] == unique_id]

            # Mask the raster with the shapes
            out_image, out_transform = mask(src, shapes.geometry, crop=True)

            # Remove the values equal to the nodata value in the raster
            no_data = src.nodata
            if raster_val_to_match is None:
                data = out_image[out_image != no_data]
            else:
                if type(raster_val_to_match) is not int:
                    try:
                        data = np.isin(out_image, raster_val_to_match)
                        count_matches = np.count_nonzero(data)
                        count_total = np.count_nonzero(out_image)
                        percent_match = count_matches / count_total * 100
                        gdf.loc[gdf[shp_id_field] == unique_id, percent_field_name] = percent_match
                    except:
                        print("The raster_val_to_match must be an integer or a list of integers")
                        return
                else:
                    data = out_image[out_image == raster_val_to_match]
            
            if data.size > 0:
                # Calculate mean and standard deviation
                mean_val = np.mean(data)
                std_dev = np.std(data)
            else:
                mean_val = np.nan
                std_dev = np.nan

            # Update the GeoDataFrame
            if mean_field_name is not None:
                gdf.loc[gdf[shp_id_field] == unique_id, mean_field_name] = mean_val
            if std_dev_field_name is not None:
                gdf.loc[gdf[shp_id_field] == unique_id, std_dev_field_name] = std_dev
            progress_counter += 1
            #print out update to progress in place that overwrites the previous line
            print(f"Progress: {progress_counter}/{len(gdf[shp_id_field].unique())}", end='\r')
        # Save the updated GeoDataFrame to a new shapefile
        gdf.to_file(output_shapefile_path)


def main():
    raster_file_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\LM2_070923\RF_Results\Stitched_Classification.tif"
    shapefile_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\LM2\Summary_Stats_JTM\Summary_Stats_JTM.shp"
    
    
    kwargs = {'shapefile_ID': 'ID', 
     'mean_field_name': None, #Set to None if you don't want to calculate mean
     'std_dev_field_name': None, #Set to None if you don't want to calculate standard deviation
     'percent_field_name': '% BE', 
     'raster_val_to_match': [4, 5], #Set to None if you don't want to calculate percent of raster values that match a certain value
     'output_shapefile_path': r'Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\LM2\Hillslope_Stats\Hillslope_Stats_ATD_pruned.shp',}
    get_raster_stats_and_update_shapefile(raster_file_path, shapefile_path, **kwargs)

if __name__ == main():
    main()