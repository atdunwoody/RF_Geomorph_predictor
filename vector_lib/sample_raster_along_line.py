"""
Samples an input raster along a line geometry and plots the distance vs raster value.
If a polygon is provided, the intersection points between the line and the polygon are marked on the plot. 
"""

import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point

def sample_raster_along_line(line, raster, n_points=100):
    # Generate n_points evenly spaced along the line without normalization
    distances = np.linspace(0, line.length, n_points)
    points = [line.interpolate(distance) for distance in distances]
    
    # Extract raster values at the points
    raster_values = []
    for point in points:
        row, col = raster.index(point.x, point.y)
        raster_values.append(raster.read(1)[row, col])
    
    return distances, raster_values

def find_intersections_with_polygon(line, polygon):
    intersections = line.intersection(polygon)
    if intersections.is_empty:
        return []
    elif isinstance(intersections, Point):
        return [line.project(intersections)]
    elif isinstance(intersections, LineString):
        return [line.project(Point(intersections.coords[0])), line.project(Point(intersections.coords[-1]))]
    else:
        return [line.project(Point(geom.coords[0])) for geom in intersections.geoms if isinstance(geom, Point)]

def plot_distance_vs_raster_value(line, raster, line_index, output_folder, polygon=None):
    distances, raster_values = sample_raster_along_line(line, raster)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, raster_values, marker='o', linestyle='-', color='b')
    plt.title(f'Line Index: {line_index} - Distance vs Raster Value')
    plt.xlabel('Distance along line')
    plt.ylabel('Raster Value')
    plt.grid(True)
    
    # Mark the intersection points with the polygon
    if polygon is not None:
        intersection_distances = find_intersections_with_polygon(line, polygon)
        for intersection_distance in intersection_distances:
            plt.axvline(x=intersection_distance, color='r', linestyle='--', label='Intersection')
        if intersection_distances:
            plt.legend()

    plt.savefig(f'{output_folder}/distance_vs_raster_{line_index}.png')
    plt.close()

def main(gpkg_path, raster_path, output_folder, polygon_path=None):
    # Load line geometries from the GeoPackage
    gdf = gpd.read_file(gpkg_path)
    
    # Load polygon if provided
    polygon = None
    if polygon_path:
        polygon_gdf = gpd.read_file(polygon_path)
        polygon = polygon_gdf.geometry.iloc[0]  # Assuming there's only one polygon
    
    # Open the raster
    with rasterio.open(raster_path) as raster:
        # Loop through each line in the GeoPackage
        for idx, row in gdf.iterrows():
            line = row.geometry
            print(f'Processing line at index {idx}')
            if isinstance(line, LineString):
                plot_distance_vs_raster_value(line, raster, idx, output_folder, polygon)
            else:
                print(f'Skipping non-LineString geometry at index {idx}')

if __name__ == "__main__":
    gpkg_path = r"Y:\ATD\GIS\ETF\Watershed Stats\Channels\Perpendiculars\LPM_perpendicular.gpkg"  # Replace with your GeoPackage path
    raster_path = r"Y:\ATD\GIS\ETF\Terrain Derivatives\Slope.tif"  # Replace with your raster path
    output_folder = r"Y:\ATD\GIS\ETF\Watershed Stats\Channels\Perpendiculars"  # Replace with your desired output folder path
    polygon_path = r"Y:\ATD\GIS\ETF\Watershed Stats\Channels\Valleys\LPM Valley.gpkg" # Replace with your polygon GeoPackage path
    
    main(gpkg_path, raster_path, output_folder, polygon_path)
