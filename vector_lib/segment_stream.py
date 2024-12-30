import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import split
import numpy as np
import rasterio
from rasterio.features import rasterize
from skimage.morphology import skeletonize
from geoprocessing_tools import multipolygon_to_polygon
import os
from tqdm import tqdm

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

def create_smooth_perpendicular_lines(centerline_path, line_length=60, spacing=5, window=20, output_path=None):
    # Load the centerline from the geopackage
    gdf = gpd.read_file(centerline_path)
    
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
    if output_path is not None:
        perpendiculars_gdf.to_file(output_path, driver='GPKG')
    return perpendiculars_gdf

def segment_stream_polygon(stream_polygon_path, centerline_path, output_path, segment_spacing=20, window=100, scaling_factor=100):
    """
    Segments a stream polygon into smaller sections using cutting lines perpendicular 
    to a centerline. The cutting lines are placed at regular intervals along the centerline, 
    and the polygon is split along these lines.

    Parameters:
    -----------
    stream_polygon_path : str
        Path to the GeoPackage or shapefile containing the stream polygon.
    
    centerline_path : str
        Path to the GeoPackage or shapefile containing the centerline geometry.
    
    output_path : str
        Path to save the segmented polygons as a new GeoPackage or shapefile.
    
    segment_spacing : int, optional
        Width of each segment of the river corridor.
    
    window : int, optional
        The distance in meters to look ahead and behind the interpolation point on 
        the centerline for averaging the direction of the perpendicular cutting line 
        (default is 20 meters).

    Returns:
    --------
    None
        The function saves the segmented polygons to the specified output file.
    """
    print(f"Segmenting stream polygon: {stream_polygon_path}")
    # Load the shapefile and the centerline
    gdf = gpd.read_file(stream_polygon_path)
    centerline_gdf = gpd.read_file(centerline_path)
    
    # Assuming the polygon to segment is the first feature in the shapefile
    polygon = gdf.geometry[0]
    centerline = centerline_gdf.geometry[0]
    
    # Set n_segments to the length of the centerline
    n_segments = int(centerline.length / segment_spacing)
    print(f"Segmenting into {n_segments} segments")
    # Calculate interval along the centerline to place cutting points
    line_length = centerline.length
    interval = line_length / n_segments
    
    # Initialize list to store cutting lines
    cutting_lines = []
    
    # tqdm progress bar added here
    for i in tqdm(range(1, n_segments), desc="Creating cutting lines"):
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
        perp_dx = -dy_avg / length_vector * scaling_factor
        perp_dy = dx_avg / length_vector * scaling_factor
        
        # Define a long perpendicular line for cutting
        start_point = Point(point.x + perp_dx, point.y + perp_dy)
        end_point = Point(point.x - perp_dx, point.y - perp_dy)
        cutting_lines.append(LineString([start_point, end_point]))
    
    # Initial set of segments
    segments = [polygon]

    # tqdm progress bar added here for splitting the polygon
    for line in tqdm(cutting_lines, desc="Splitting polygons"):
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

def segment_channel_directory(centerline_dir, channel_poly_dir, output_segments_dir):
    # centerline_dir = r"Y:\ATD\GIS\ETF\Watershed Stats\Channels\Centerlines"
    # input_dir = r"Y:\ATD\GIS\Bennett\Channel Polygons"

    # output_segment_dir = r"Y:\ATD\GIS\ETF\Watershed Stats\Channels\Perpendiculars"
    if not os.path.exists(output_segments_dir):
        os.makedirs(output_segments_dir)
    watersheds = ['LM2', 'LPM', 'MM', 'MPM', 'UM1', 'UM2']
    
    for watershed in watersheds:
        #search for right raster by matching the watershed name
        for file in os.listdir(channel_poly_dir):
            if watershed in file and file.endswith('.gpkg'):
                input_path = os.path.join(channel_poly_dir, file)
                print(f"Input: {input_path}")
                break

        centerline_path = os.path.join(centerline_dir, f'{watershed} centerline.gpkg')
        output_segment_path = os.path.join(output_segments_dir, f'{watershed}_channel_segmented.gpkg')
        print(f"Processing watershed: {watershed}")
        
        print(f"Centerline: {centerline_path}")
        print(f"Output: {output_segment_path}\n")
        #create_centerline(input_path, centerline_path)
        #multipolygon_to_polygon(chan_path, output_path)
        #segment_stream_polygon(input_path, centerline_path, output_segment_path, segment_spacing = 20)
        perp_lines = create_smooth_perpendicular_lines(centerline_path, line_length=60, spacing=500, window=10)
        
        out_perp_path = os.path.join(output_segments_dir, f'{watershed}_perpendicular.gpkg')
        perp_lines.to_file(out_perp_path, driver='GPKG')

def main():

    input_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leving Test\Bennett_Centerlines_EPSG26913_buffered.gpkg"
    centerline_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Channel Polygons\Centerlines_LSDTopo\Bennett_Centerlines_EPSG26913_single.gpkg"
    output_segment_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leving Test\Bennett_Centerlines_EPSG26913_5m.gpkg"
    output_perp_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley_Footprints\Hydraulic Model\Depth Leving Test\Bennett_Centerlines_EPSG26913_perpendiculars.gpkg"
    create_smooth_perpendicular_lines(centerline_path, line_length=110, spacing=5, window=100, output_path=output_perp_path)
    #segment_stream_polygon(input_path, centerline_path, output_segment_path, segment_spacing = 5, scaling_factor=120)
    
if __name__ == '__main__':
    main()
