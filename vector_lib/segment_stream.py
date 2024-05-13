import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import split
import numpy as np
import rasterio
from rasterio.features import rasterize
from skimage.morphology import skeletonize
from geoprocessing_tools import multipolygon_to_polygon

def create_centerline(polygon_file, output_CL_file):
    """Process each polygon in the GeoPackage to create centerlines and save them."""
    def polygon_to_raster(poly, cell_size=1):
        """Convert a polygon to a raster array."""
        bounds = poly.bounds
        width = int(np.ceil((bounds[2] - bounds[0]) / cell_size))
        height = int(np.ceil((bounds[3] - bounds[1]) / cell_size))
        transform = rasterio.transform.from_origin(bounds[0], bounds[3], cell_size, cell_size)
        raster = rasterize([(poly, 1)], out_shape=(height, width), transform=transform)
        return raster, transform

    def raster_to_centerline(raster, transform):
        """Convert raster array to a centerline geometry."""
        skeleton = skeletonize(raster == 1)
        points = [Point(*rasterio.transform.xy(transform, row, col, offset='center'))
                for row in range(skeleton.shape[0]) for col in range(skeleton.shape[1]) if skeleton[row, col]]
        if points:
            line = LineString(points)
            return line
        return None

    def calc_centerline(polygon, cell_size=1):
        """Main function to create centerline from a polygon."""
        raster, transform = polygon_to_raster(polygon, cell_size)
        centerline = raster_to_centerline(raster, transform)
        return centerline
    

    gdf = gpd.read_file(polygon_file)
    gdf['centerline'] = gdf['geometry'].apply(lambda x: calc_centerline(x))

    # Remove entries where no centerline was found
    centerlines_gdf = gdf.dropna(subset=['centerline'])
    centerlines_gdf = centerlines_gdf.set_geometry('centerline', drop=True)  # Set 'centerline' as the geometry column and drop the old one
    centerlines_gdf.crs = gdf.crs  # Ensure CRS is preserved
    # Save to a new GeoPackage
    centerlines_gdf.to_file(output_CL_file, layer='centerlines', driver='GPKG')

    return centerlines_gdf

def create_perpendicular_lines(gpkg_path, distance=100, spacing=1):
    # Load the centerline from the geopackage
    gdf = gpd.read_file(gpkg_path)
    
    # Initialize an empty list to store perpendicular lines
    perpendiculars = []
    
    # Iterate through each feature in the GeoDataFrame
    for _, row in gdf.iterrows():
        geometry = row['geometry']
        
        # Handle MultiLineString appropriately using `geoms`
        if isinstance(geometry, MultiLineString):
            line_parts = geometry.geoms
        else:
            line_parts = [geometry]

        # Process each line part
        for line in line_parts:
            coords = np.array(line.coords)
            for i in range(0, len(coords) - 1, spacing):  # Adjust spacing here
                p1, p2 = coords[i], coords[i+1]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                
                # Calculate the perpendicular vector
                len_vector = np.sqrt(dx**2 + dy**2)
                perp_vector = (-dy, dx)
                
                # Normalize and scale the vector
                perp_vector = (perp_vector[0] / len_vector * distance, perp_vector[1] / len_vector * distance)
                
                # Calculate mid-point of the line segment
                mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                
                # Create the perpendicular line segment
                perp_line = LineString([
                    (mid_point[0] + perp_vector[0], mid_point[1] + perp_vector[1]),
                    (mid_point[0] - perp_vector[0], mid_point[1] - perp_vector[1])
                ])
                
                # Append the perpendicular line to the list
                perpendiculars.append({'geometry': perp_line})
    
    # Convert list to GeoDataFrame
    perpendiculars_gdf = gpd.GeoDataFrame(perpendiculars, crs=gdf.crs)
    
    # Save the perpendicular lines to the same geopackage
    out_gpkg_path = gpkg_path.replace('.gpkg', '_perpendiculars.gpkg')
    perpendiculars_gdf.to_file(out_gpkg_path, driver='GPKG')

def create_smooth_perpendicular_lines(gpkg_path, line_length=60, spacing=5, window=20):
    # Load the centerline from the geopackage
    gdf = gpd.read_file(gpkg_path)
    
    # Initialize an empty list to store perpendicular lines
    perpendiculars = []
    
    # Iterate through each feature in the GeoDataFrame
    for _, row in gdf.iterrows():
        geometry = row['geometry']
        
        # Handle MultiLineString appropriately using `geoms`
        if isinstance(geometry, MultiLineString):
            line_parts = geometry.geoms
        else:
            line_parts = [geometry]

        # Process each line part
        for line in line_parts:
            length = line.length
            num_samples = int(np.floor(length / spacing))
            for i in range(num_samples + 1):
                # Calculate the point at each meter
                point = line.interpolate(i * spacing)
                
                # Get points 20 meters ahead and behind
                point_back = line.interpolate(max(0, i * spacing - window))
                point_forward = line.interpolate(min(length, i * spacing + window))
                
                # Calculate vectors to these points
                dx_back, dy_back = point.x - point_back.x, point.y - point_back.y
                dx_forward, dy_forward = point_forward.x - point.x, point_forward.y - point.y
                
                # Average the vectors
                dx_avg = (dx_back + dx_forward) / 2
                dy_avg = (dy_back + dy_forward) / 2
                
                # Calculate the perpendicular vector
                len_vector = np.sqrt(dx_avg**2 + dy_avg**2)
                perp_vector = (-dy_avg, dx_avg)
                
                # Normalize and scale the vector
                perp_vector = (perp_vector[0] / len_vector * line_length, perp_vector[1] / len_vector * line_length)
                
                # Create the perpendicular line segment
                perp_line = LineString([
                    (point.x + perp_vector[0], point.y + perp_vector[1]),
                    (point.x - perp_vector[0], point.y - perp_vector[1])
                ])
                
                # Append the perpendicular line to the list
                perpendiculars.append({'geometry': perp_line})
    
    # Convert list to GeoDataFrame
    perpendiculars_gdf = gpd.GeoDataFrame(perpendiculars, crs=gdf.crs)
    
    # Save the perpendicular lines to the same geopackage
    out_gpkg_path = gpkg_path.replace('.gpkg', '_perpendiculars_100m.gpkg')
    perpendiculars_gdf.to_file(out_gpkg_path, driver='GPKG')

def segment_stream_polygon(stream_polygon_path, centerline_path, output_path, n_segments = 200, window=20):
    # Load the shapefile and the centerline
    gdf = gpd.read_file(stream_polygon_path)
    centerline_gdf = gpd.read_file(centerline_path)
    
    # Assuming the polygon to segment is the first feature in the shapefile
    polygon = gdf.geometry[0]
    centerline = centerline_gdf.geometry[0]
    
    # Calculate interval along the centerline to place cutting points
    line_length = centerline.length
    interval = line_length / n_segments
    
    # Initialize list to store cutting lines
    cutting_lines = []
    
    for i in range(1, n_segments):
        # Calculate the primary interpolation point
        point = centerline.interpolate(i * interval)

        # Calculate points 20 meters behind and ahead for rolling average
        point_back = centerline.interpolate(max(0, i * interval - window))
        point_forward = centerline.interpolate(min(line_length, i * interval + window))
        
        # Determine vectors to these points
        dx_back, dy_back = point.x - point_back.x, point.y - point_back.y
        dx_forward, dy_forward = point_forward.x - point.x, point_forward.y - point.y
        
        # Average the vectors
        dx_avg = (dx_back + dx_forward) / 2
        dy_avg = (dy_back + dy_forward) / 2
        
        # Compute the perpendicular vector
        length_vector = np.sqrt(dx_avg**2 + dy_avg**2)
        perp_dx = -dy_avg / length_vector
        perp_dy = dx_avg / length_vector
        
        # Define a long perpendicular line for cutting
        start_point = Point(point.x + perp_dx * 1000, point.y + perp_dy * 1000)
        end_point = Point(point.x - perp_dx * 1000, point.y - perp_dy * 1000)
        cutting_lines.append(LineString([start_point, end_point]))
    
    # Initial set of segments
    segments = [polygon]

    # Split the polygon with each line
    for line in cutting_lines:
        new_segments = []
        for segment in segments:
            split_result = split(segment, line)
            new_segments.extend(split_result.geoms)
        segments = new_segments

    # Convert segments to GeoDataFrame
    segment_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(segments))

    # Set the same CRS as the original
    segment_gdf.crs = gdf.crs

    # Save to a new shapefile
    segment_gdf.to_file(output_path)
    
def main():
    chan_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Test - Slope\LM2 Channel Stats.gpkg"
    output_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Test - Slope\Buffer as Lines\Test Single Poly.gpkg"
    output_segment_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Test - Slope\Buffer as Lines\Test Single Poly Segmented.gpkg"
    centerline_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Test - Slope\LM2 Centerline.gpkg"
    multipolygon_to_polygon(chan_path, output_path)
    segment_stream_polygon(output_path, centerline_path, output_segment_path)
    
if __name__ == '__main__':
    main()