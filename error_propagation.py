import raster_lib.raster_utils as ru
import numpy as np

def gross_error_propagation(raster_list):
    for raster in raster_list:
        outfile = raster.split(".")[0] + "_LIDAR_error_prop.tif"
        #change_raster_crs(raster, outfile, crs)
        #0.100 is LIDAR RMSEz reported for 2020 LIDAR flights
        ru.apply_math_to_raster(raster, outfile, lambda x: 1.96 * np.sqrt(0.100**2 + (x/1000)**2))