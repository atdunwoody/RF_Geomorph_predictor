from osgeo import gdal
import numpy as np
from raster_utils import save_dem

def compute_slope(dem_path, output_path):
    """Compute the slope of a DEM raster and save the output."""
    dem = gdal.Open(dem_path)
    gdal.DEMProcessing(output_path, dem, 'slope')
    dem = None  # Close the dataset
    
def compute_aspect(dem_path, output_path):
    """Compute the aspect of a DEM raster and save the output."""
    dem = gdal.Open(dem_path)
    gdal.DEMProcessing(output_path, dem, 'aspect')
    dem = None  # Close the dataset

def compute_curvature(dem_path, output_path):
    """Compute the curvature of a DEM raster and save the output."""
    dem = gdal.Open(dem_path)
    gdal.DEMProcessing(output_path, dem, 'curvature')
    dem = None  # Close the dataset
    
def compute_twi(flow_acc_path, slope_path, output_path):
    """Compute Topographic Wetness Index and save the output raster."""
    slope = gdal.Open(slope_path)
    slope_band = slope.GetRasterBand(1).ReadAsArray()
    slope_radians = np.radians(slope_band)
    flow_acc = gdal.Open(flow_acc_path)
    flow_acc_band = flow_acc.GetRasterBand(1).ReadAsArray()
    with np.errstate(divide='ignore', invalid='ignore'):
        twi = np.log((flow_acc_band + 1) / np.tan(slope_radians))
        twi[np.isinf(twi)] = 0  # Replace infinities
    save_dem(twi, slope_path, output_path)

def compute_roughness(dem_path, output_path):
    """Compute Roughness and save the output raster."""
    dem = gdal.Open(dem_path)
    dem_band = dem.GetRasterBand(1).ReadAsArray()
    kernel_size = 3
    pad_width = kernel_size // 2
    padded_dem = np.pad(dem_band, pad_width, mode='reflect')
    shape = (dem_band.shape[0], dem_band.shape[1], kernel_size, kernel_size)
    strides = padded_dem.strides[:2] + padded_dem.strides
    windows = np.lib.stride_tricks.as_strided(padded_dem, shape=shape, strides=strides)
    roughness = np.std(windows, axis=(2, 3))
    save_dem(roughness, dem_path, output_path)

def compute_ruggedness(dem_path, output_path):
    """Compute Ruggedness (TRI) and save the output raster."""
    dem = gdal.Open(dem_path)
    dem_band = dem.GetRasterBand(1).ReadAsArray()
    kernel_size = 3
    pad_width = kernel_size // 2
    padded_dem = np.pad(dem_band, pad_width, mode='reflect')
    center = padded_dem[1:-1, 1:-1]
    ruggedness = np.sum(np.abs(padded_dem[:-2, :-2] - center))
    ruggedness += np.sum(np.abs(padded_dem[:-2, 1:-1] - center))
    ruggedness += np.sum(np.abs(padded_dem[:-2, 2:] - center))
    ruggedness += np.sum(np.abs(padded_dem[1:-1, :-2] - center))
    ruggedness += np.sum(np.abs(padded_dem[1:-1, 2:] - center))
    ruggedness += np.sum(np.abs(padded_dem[2:, :-2] - center))
    ruggedness += np.sum(np.abs(padded_dem[2:, 1:-1] - center))
    ruggedness += np.sum(np.abs(padded_dem[2:, 2:] - center))
    save_dem(ruggedness, dem_path, output_path)
