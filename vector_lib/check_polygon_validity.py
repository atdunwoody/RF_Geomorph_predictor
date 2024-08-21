import geopandas as gpd

def main():
    input_gpkg = r"Y:\ATD\GIS\Bennett\Valley Widths\Bennett_Channels.gpkg"
    output_gpkg = r"Y:\ATD\GIS\Bennett\Valley Widths\Bennett_Channels_Clean.gpkg"
    clip_gpkg = r"Y:\ATD\GIS\Bennett\Bennett_watersheds.gpkg"

    target_gdf = gpd.read_file(input_gpkg)
    clipper_gdf = gpd.read_file(clip_gpkg)

    # Fix invalid geometries
    target_gdf['geometry'] = target_gdf['geometry'].buffer(0)
    clipper_gdf['geometry'] = clipper_gdf['geometry'].buffer(0)

    # Check if geometries are valid
    if target_gdf.is_valid.all() and clipper_gdf.is_valid.all():
        clipped_gdf = gpd.overlay(target_gdf, clipper_gdf, how='intersection')
        # Save the result to a new GeoPackage
        clipped_gdf.to_file(output_gpkg, driver="GPKG")
    else:
        print("Some geometries are still invalid. Manual inspection is required.")

if __name__ == "__main__":
    main()

