import geopandas as gpd
from shapely.geometry import Polygon,MultiPolygon
from shapely.ops import unary_union, cascaded_union
from shapely.geometry import box

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

def multipart_poly_to_single(gdf, method='union'):
    """
    Converts all multipart polygons in a GeoDataFrame into single-part polygons,
    using either the convex hull or a union of all parts.

    Parameters:
    - gdf (GeoDataFrame): A GeoDataFrame containing the geometries to process.
    - method (str): Method to use for conversion; 'convex' for convex hull, 'union' for exact union.

    Returns:
    - GeoDataFrame: A new GeoDataFrame with all multipart polygons converted to single-part polygons.
    """
    def to_single_polygon(geometry):
        # Check if the geometry is None
        if geometry is None:
            return None  # Return None if the geometry is None
        
        if geometry.geom_type == 'MultiPolygon':
            if method == 'convex':
                # Use the convex hull of all parts
                return geometry.convex_hull
            elif method == 'union':
                # Use a precise union of all parts
                return unary_union(geometry)
        elif geometry.geom_type == 'Polygon':
            return geometry
        else:
            return geometry  # Return non-polygon geometries unchanged

    # Apply the to_single_polygon function to each geometry in the GeoDataFrame
    gdf['geometry'] = gdf['geometry'].apply(to_single_polygon)
    return gdf

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

# Load your GeoDataFrames
target_fn = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\LPM\Catchment Delineation\Summary_Stats_JTM_cleaned.shp"
eraser_fn = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\LPM\Channel Delineation\LPM_Channel_10m_Section.shp"
target_layer = gpd.read_file(target_fn)
eraser_layer = gpd.read_file(eraser_fn)
out_file = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\LPM\Catchment Delineation\Summary_Stats_JTM_filled_v2.shp"
# Perform the erase operation
#erased_layer = erase(target_layer, eraser_layer)
filled_holes = fill_polygon_holes(target_layer, out_file)
# Optionally, save the result to a new shapefile
#filled_holes.to_file(r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\LPM\Catchment Delineation\Summary_Stats_JTM_filled_v2.shp")
