import os 

mi60_raster = r"Y:\ATD\GIS\MRMS_Data\Summary Data\20230709-20220812\MRMS_MI60_20220812-20230709_UTM.tif"
dem_raster = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\ET_low_LIDAR_2020_1m_DEM.tif"
flow_accumulation_raster = r"Y:\ATD\GIS\ETF\DEMs\LIDAR\OT 2020\WBT_Outputs_Low\flow_accumulation.tif"
sbs_raster = r"Y:\ATD\GIS\ETF\dNBR\east_troublesome_co4020310623920201014_sbs.tif"
land_cover_raster = r"Y:\ATD\GIS\ETF\Vegetation Filtering\LM2\LM2_081222\RF_Results\Stitched_Classification.tif"
DoD_raster = r"Y:\ATD\GIS\ETF\DEMs\SfM\LM2\LM2_2023 Exports\DoD\DoD DoD LM2_2023____070923_PostError_PCFiltered_DEM - LM2_2023____081222_PostError_PCFiltered_DEM.tif"
accumulation_raster = r"Y:\ATD\GIS\MRMS_Data\Summary Data\20230709-20220812\MRMS_accum_20220812-20230709_UTM.tif"

field_map = {
    dem_raster: ['DEM', 'elevation'],
    flow_accumulation_raster: ['flow_accum'],
    mi60_raster: ['mi60'],
    accumulation_raster: ['accum_precip'],
    sbs_raster: ['sbs'],
    land_cover_raster: ['landcover', 'bare_earth'],
    DoD_raster: ['net_change', 'erosion', 'deposition']
}

def build_params(field_name = 'erosion', stats = ['mean'], watershed = 'LM2'):
    stats = [stat.lower() for stat in stats]
    threshold = None
    threshold_direction = None
    match_value = None
    raster_path = None
    #get the field name from the field_map
    for key, value in field_map.items():
        for val in value:
            if val in field_name.lower():
                raster_path = key
                break

    #check if raster path ends with a directory or a file
    if raster_path is not None:
        if os.path.isdir(raster_path):
            #search for right raster by matching the watershed name
            for file in os.listdir(raster_path):
                if watershed in file and file.endswith('.tif'):
                    raster_path = os.path.join(raster_path, file)
                    break
            
    stat_key = {}
    for stat in stats:
        #Capitalize the first letter of each word
        stat = ' '.join([word.capitalize() for word in stat.split()])
        field_name = ' '.join([word.capitalize() for word in field_name.split()])   
        stat_key[stat] = f'{field_name} {stat}'
     
    if 'erosion' in field_name.lower() or 'sfm erosion' in field_name.lower():
        threshold = 0
        threshold_direction = 'below'
        
    if 'deposition' in field_name.lower() or 'sfm deposition' in field_name.lower():
        threshold = 0
        threshold_direction = 'above'
        
    if 'veg' in field_name.lower():
        match_value = [1, 2, 3]
    
    if 'BE' in field_name.lower():
        match_value = [4,5]
    
    if 'log' in field_name.lower():
        match_value = [2, 3]
    
    params= {
            'raster_path': raster_path,
            'threshold': threshold,
            'threshold_direction': threshold_direction,
            'raster_value_to_match': match_value,
            'stat_key': stat_key
        }
    return params
                