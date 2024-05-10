import geopandas as gpd
import numpy as np
from osgeo import gdal
from shapely.geometry import LineString, Point
import rasterio
from rasterio.features import rasterize
from compute_hillslope_attributes import aggregate_erosion_deposition, aggregate_masked_erosion_deposition

def overlay_points_with_buffer_on_raster(points_gpkg_path, raster_path, buffer_size, output_gpkg_path):
    # Load the points from the geopackage
    points_gdf = gpd.read_file(points_gpkg_path)

    # Open the raster file
    with rasterio.open(raster_path) as raster:
        # Coordinate transformation from geopandas GeoDataFrame to raster coordinates
        raster_affine = raster.transform
        raster_crs = raster.crs

        # Reproject points GeoDataFrame to the raster's CRS
        points_gdf = points_gdf.to_crs(raster_crs)

        # Buffer points in the coordinate system of the raster
        points_gdf['geometry'] = points_gdf.geometry.buffer(buffer_size, resolution=16)

        # Extract raster values within the buffer
        values = []
        for geom in points_gdf.geometry:
            # Rasterize the buffered geometry to match the raster
            mask = rasterize([(geom, 1)], out_shape=raster.shape, transform=raster.transform, fill=0, all_touched=True, dtype='uint8')
            # Extract data within the mask
            data = raster.read(1)
            masked_data = data[mask == 1]

            # Compute the mean of the masked data
            if masked_data.size > 0:
                mean_value = masked_data.mean()
            else:
                mean_value = np.nan  # In case the mask does not overlap any raster cells
            print(mean_value)
            values.append(mean_value)

        # Add raster values to the GeoDataFrame
        points_gdf['raster_mean_value'] = values

    # Save the updated GeoDataFrame back to a new geopackage
    points_gdf.to_file(output_gpkg_path, driver='GPKG')

    print("Updated GeoPackage saved to:", output_gpkg_path)
    return points_gdf
    
def create_points_along_path(filepath):
    # Load the data from the geopackage
    gdf = gpd.read_file(filepath)
    
    # Create a LineString from the Point geometries
    line = LineString([point for point in gdf.geometry])

    # Create points along the generated line at 1-meter intervals
    length = line.length
    distances = np.arange(0, length + 1, 1)
    points = [line.interpolate(distance) for distance in distances]

    # Calculate angles and cumulative distance
    angles = [0]  # Start with 0 angle for the first point
    total_distances = [0]  # Starting point, so distance is 0

    # Calculate cumulative distances for each point
    for i in range(1, len(points)):
        # Update cumulative distance
        segment_length = points[i].distance(points[i-1])
        total_distances.append(total_distances[-1] + segment_length)

    # Calculate the internal angles formed at each point
    for i in range(1, len(points)-1):
        # Vectors for calculating angle
        vector_a = np.array([points[i].x - points[i-1].x, points[i].y - points[i-1].y])
        vector_b = np.array([points[i+1].x - points[i].x, points[i+1].y - points[i].y])

        # Normalize vectors to prevent division by zero
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            # One or both vectors are zero vectors, so angle is undefined; set to 0
            angle_deg = 0
        else:
            # Calculate angle using the dot product and magnitude of vectors
            cos_angle = np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angle_deg = np.degrees(angle)  # Convert to degrees
        
        angles.append(angle_deg)
        
    # Set angle for the last point
    angles.append(0)  # No angle at the last point due to absence of a subsequent point

    # Create a GeoDataFrame from the list of points
    points_gdf = gpd.GeoDataFrame({
        'geometry': points,
        'angle': angles,
        'total_distance': total_distances
    }, crs=gdf.crs)
    
    out_prefix = filepath.split(".")[0][0:3]
    outfile = out_prefix + " 1m points.gpkg"
    points_gdf.to_file(outfile, driver="GPKG")
    return points_gdf


def main():
    # Define the filepath
    filepath = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Channels\Points along Channel\LPM 5m points.gpkg"

    # Generate points along the reconstructed line from the points
    # points_along_line = create_points_along_path(filepath)
    
    points_gpkg_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Channels\Points along Channel\LM2 1m points.gpkg"
    raster_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Terrain Feature Rasters\Slope.tif"
    output_gpkg_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Channels\Points along Channel\LM2 1m points.gpkg"

    # points_gdf = overlay_points_with_buffer_on_raster(points_gpkg_path, raster_path, 0.05, output_gpkg_path)

    channel_list  = [
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Channels\LM2 Centerline.gpkg",
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Channels\LPM Centerline.gpkg",
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Channels\MM Centerline.gpkg",
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Channels\MPM Centerline.gpkg",
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Channels\UM1 Centerline.gpkg",
                    r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Channels\UM2 Centerline.gpkg"
                    ]
    
    aggregate_erosion_deposition(channel_list, 0.05)
    aggregate_masked_erosion_deposition(channel_list, 0.05)
    
if __name__ == '__main__':
    main()