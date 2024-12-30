import geopandas as gpd

def get_valley_widths(valley_footprint_gpkg, channel_segments_gpkg, perpendicular_lines_gpkg):
    """
    Calculate the average valley width by clipping perpendicular lines with the valley footprint
    and associating them with channel segments.

    Parameters:
    - valley_footprint_gpkg: str, path to the GeoPackage containing the valley footprint polygon.
    - channel_segments_gpkg: str, path to the GeoPackage containing the channel segments multipolygon.
    - perpendicular_lines_gpkg: str, path to the GeoPackage containing perpendicular lines.

    The function updates the 'channel_segments_gpkg' by adding a new field 'valley_width' representing
    the average length of the clipped perpendicular lines associated with each channel segment.
    """
    # Read in the data
    valley_footprint = gpd.read_file(valley_footprint_gpkg)
    channel_segments = gpd.read_file(channel_segments_gpkg)
    perpendicular_lines = gpd.read_file(perpendicular_lines_gpkg)
    
    # Ensure all layers are in the same CRS
    valley_footprint = valley_footprint.to_crs(perpendicular_lines.crs)
    channel_segments = channel_segments.to_crs(perpendicular_lines.crs)
    
    # Clip the perpendicular_lines by the valley_footprint
    clipped_lines = gpd.clip(perpendicular_lines, valley_footprint)
    
    # Compute the length of each clipped line
    clipped_lines['length'] = clipped_lines.length
    
    # Spatial join between clipped lines and channel_segments
    # This will associate each line with the channel segment it intersects
    clipped_lines_with_segments = gpd.sjoin(clipped_lines, channel_segments, how='left', predicate='intersects')
    
    # Now, group by the index of the channel_segments and compute average length
    # The 'index_right' column contains the index of the channel segment
    average_lengths = clipped_lines_with_segments.groupby('index_right')['length'].mean()
    
    # Now, add the average lengths to the channel_segments GeoDataFrame
    channel_segments['valley_width'] = channel_segments.index.map(average_lengths)
    
    # Handle NaN values if any channel segment did not get associated with any clipped lines
    channel_segments['valley_width'] = channel_segments['valley_width'].fillna(0)
    
    # Save the updated channel_segments to the gpkg
    channel_segments.to_file(channel_segments_gpkg, driver='GPKG')
    return channel_segments_gpkg

if __name__ == '__main__':
    valley_footprint_gpkg = r"Y:\ATD\GIS\ETF\Valley Bottoms\ATD_Algorithm\LM2_valleys_wavelets.gpkg"
    channel_segments_gpkg = r"Y:\ATD\GIS\ETF\Watershed Stats\Test\LM2_channel_ordered_segments.gpkg"
    perpendicular_lines_gpkg = r"Y:\ATD\GIS\ETF\Watershed Stats\Test\LM2_channel_perpendiculars.gpkg"
    get_valley_widths(valley_footprint_gpkg, channel_segments_gpkg, perpendicular_lines_gpkg)