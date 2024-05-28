import os 

DEM_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LIDAR\Reprojected to UTM Zone 13N\ET_merged_LIDAR_2020_1m_DEM_reproj.tif"
DoD_dir = r"Y:\ATD\DEM_Alignment\East_Troublesome_Alignment\DoD 070923\Vegetation Masked"
error_DoD_dir = r"Y:\ATD\DEM_Alignment\East_Troublesome_Alignment\DoD 070923\Vegetation Masked\Error Thresholded_10cm"
SfM_DoD_dir = r"Y:\ATD\DEM_Alignment\East_Troublesome_Alignment\DoD 070923 SfM"
flow_accum_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Terrain Feature Rasters\Flow Accumulation.tif"
slope_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Terrain Feature Rasters\Slope.tif"
aspect_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Terrain Feature Rasters\Aspect.tif"
dNBR_path = r"Y:\ATD\GIS\East_Troublesome\Watershed Statistical Analysis\Other Features\dnbr_20200904_20210907.tif"
classification_dir = r"Y:\ATD\DEM_Alignment\East_Troublesome_Alignment\DoD 070923\RF Land Cover Classifications"

field_map = {
    DEM_path: ['DEM', 'elevation'],
    DoD_dir: ['net change'],
    error_DoD_dir: ['erosion', 'deposition', ],
    SfM_DoD_dir: ['sfm erosion', 'sfm deposition'],
    flow_accum_path: ['flow'],
    classification_dir: ['log', 'veg', 'BE'],
    slope_path: ['slope'],
    aspect_path: ['aspect'],
    dNBR_path: ['dNBR']
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
                