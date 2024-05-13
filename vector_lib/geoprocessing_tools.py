import geopandas as gpd
from shapely.geometry import Polygon,MultiPolygon
from shapely.ops import unary_union, cascaded_union
from shapely.geometry import box
from shapely.geometry import LineString, MultiLineString
import numpy as np
from shapely.geometry import shape, mapping
import fiona

def erase(target_gdf, eraser_gdf):
    """
    Performs a geometric erase operation on the target GeoDataFrame using the eraser GeoDataFrame.

    Parameters:
    - target_gdf (GeoDataFrame): The GeoDataFrame to be erased.
    - eraser_gdf (GeoDataFrame): The GeoDataFrame that defines the areas to erase.

    Returns:
    - GeoDataFrame: The result of erasing the specified areas from the target GeoDataFrame.
    """
    # Ensure that the data is in the same projection
    if target_gdf.crs != eraser_gdf.crs:
        eraser_gdf = eraser_gdf.to_crs(target_gdf.crs)

    # Use overlay with the difference operation
    result_gdf = gpd.overlay(target_gdf, eraser_gdf, how='difference')

    # Handle the possibility of multi-part polygons by converting them to single-part
    result_gdf['geometry'] = result_gdf['geometry'].apply(lambda x: MultiPolygon([x]) if not x.is_valid else x)

    return result_gdf

def clip(target_gdf, clipper_gdf):
    """
    Performs a geometric clip operation where the target GeoDataFrame is clipped to the boundaries of the clipper GeoDataFrame.

    Parameters:
    - target_gdf (GeoDataFrame): The GeoDataFrame that will be clipped.
    - clipper_gdf (GeoDataFrame): The GeoDataFrame that defines the clip boundaries.

    Returns:
    - GeoDataFrame: The resulting GeoDataFrame after clipping.
    """
    # Ensure that both GeoDataFrames are in the same projection
    if target_gdf.crs != clipper_gdf.crs:
        clipper_gdf = clipper_gdf.to_crs(target_gdf.crs)
    
    # Perform the clip operation using spatial join and intersection
    clipped_gdf = gpd.overlay(target_gdf, clipper_gdf, how='intersection')
    
    return clipped_gdf

def multipolygon_to_polygon(input_geopackage, output_geopackage):
    """
    Convert a multipolygon from a GeoPackage to its exterior outline and save the result to another GeoPackage.

    Args:
        input_geopackage (str): Path to the input GeoPackage.
        layer_name (str): Layer name of the multipolygon in the input GeoPackage.
        output_geopackage (str): Path to the output GeoPackage.
        output_layer_name (str): Layer name for the output polygon in the output GeoPackage.
    """
    # Read the multipolygon from the GeoPackage
    gdf = gpd.read_file(input_geopackage)
    
    # Ensure the geometry is a multipolygon
    if not all(gdf.geometry.type.isin(['MultiPolygon', 'Polygon'])):
        raise ValueError("All geometries must be (multi)polygons")
    
    # Use unary_union to dissolve all polygons into a single outline
    outline = unary_union(gdf.geometry)

    # Create a new GeoDataFrame with the resulting outline
    result_gdf = gpd.GeoDataFrame(geometry=[outline], crs=gdf.crs)

    # Write the result to the output GeoPackage
    result_gdf.to_file(output_geopackage, driver='GPKG')
    return result_gdf

def fill_holes(gdf):
    """
    Fills all holes in polygons or multipolygons in a GeoDataFrame, making each geometry solid.

    Parameters:
    - gdf (GeoDataFrame): A GeoDataFrame containing the geometries to process.

    Returns:
    - GeoDataFrame: A new GeoDataFrame with all holes removed from the geometries.
    """
    def remove_holes(geometry):
        # Check if the geometry is None
        if geometry is None:
            return None
        
        if geometry.geom_type == 'Polygon':
            return Polygon(geometry.exterior)
        
        elif geometry.geom_type == 'MultiPolygon':
            # Use the .geoms property as per Shapely 1.8+ and avoid direct iteration
            return MultiPolygon([Polygon(poly.exterior) for poly in geometry.geoms])
        
        else:
            return geometry  # Return non-polygon geometries unchanged

    # Apply the remove_holes function to each geometry in the GeoDataFrame
    gdf['geometry'] = gdf['geometry'].apply(remove_holes)
    return gdf

def clean_shp(gdf):
    """
    Cleans a GeoDataFrame by:
    - Removing rows with None geometries.
    - Removing rows with invalid geometries and attempting to fix them.
    - Removing duplicate geometries.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame to clean.

    Returns:
    - GeoDataFrame: A cleaned GeoDataFrame.
    """
    # Remove rows where geometry is None
    gdf = gdf[gdf['geometry'].notnull()]

    # Fix invalid geometries if possible, remove if not fixable
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

    # Remove duplicates based on geometries
    gdf = gdf.drop_duplicates(subset='geometry')

    return gdf

def fill_polygon_holes(input_gdf, output_file, dissolve_by=None):
    """
    Fills holes in the interior of polygons.

    Parameters:
    - input_gdf: Input GeoDataFrame containing polygons.
    - output_file: Path where the modified GeoDataFrame will be saved.
    """
    if isinstance(input_gdf, str):
        input_gdf = gpd.read_file(input_gdf)
    def fill_holes(geometry):
        if geometry.geom_type == 'Polygon':
            return Polygon(geometry.exterior)
        elif geometry.geom_type == 'MultiPolygon':
            return MultiPolygon([Polygon(poly.exterior) for poly in geometry.geoms])
        return geometry

    # Fill holes in the polygons
    filled_polygons = input_gdf.copy()
    filled_polygons['geometry'] = filled_polygons['geometry'].apply(fill_holes)
    
    # Dissolve the polygons
    if dissolve_by is not None:
        dissolved_polygons = filled_polygons.dissolve(by=dissolve_by)
    else:
        dissolved_polygons = filled_polygons.dissolve()
    
    # Save the result to a new file
    dissolved_polygons.to_file(output_file)

def create_buffer(gpkg_path, buffer_distance):
    # Open the GeoPackage with fiona to read the layer
    with fiona.open(gpkg_path) as src:
        # Schema of the new GeoPackage (adding buffered geometries)
        schema = src.schema.copy()
        schema['geometry'] = 'Polygon'  # Update geometry type if necessary
        
        # Create a new GeoPackage for output
        output_path = gpkg_path.replace('.gpkg', '_shapely_buffered.gpkg')
        with fiona.open(output_path, 'w', driver='GPKG', schema=schema, crs=src.crs) as dst:
            # Iterate over all records in the source layer, buffer them, and write to the new file
            for feature in src:
                geom = shape(feature['geometry'])
                buffered_geom = geom.buffer(buffer_distance)
                
                # Create new feature with the buffered geometry
                new_feature = {
                    'geometry': mapping(buffered_geom),
                    'properties': feature['properties']
                }
                
                dst.write(new_feature)
    
    return output_path


def main():

    chan_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Test - Slope\LM2 Channel Stats.gpkg"
    buffer_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Test - Slope\LM2 Centerline_shapely_buffered.gpkg"
    CL_path = r'Y:\\ATD\\GIS\\East_Troublesome\\Watershed Statistical Analysis\\Watershed Stats\\Test - Slope\\LM2 Centerline.gpkg'
    perp_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Test - Slope\Buffer as Lines\LM2 Centerline_perpendiculars_100m.gpkg"
    output_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Test - Slope\Chan Single.gpkg"
    section_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Test - Slope\LM2 Centerline_shapely_buffered_multipolygon.gpkg"
    buffer_outline_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Watershed Stats\Test - Slope\Buffer as Lines\Buffer outline.gpkg"
    
    
    
    #multi_poly_path = centerline_to_multipolygon(buffer_path, CL_path, length=10, width=200)
    buffer_gdf = gpd.read_file(buffer_outline_path)
    perp_gdf = gpd.read_file(perp_path)
    #clipped_gdf  = clip(perp_gdf, buffer_gdf)
    #clipped_gdf.to_file(output_path, driver='GPKG')


if __name__ == "__main__":
    main()