import rasterio
import numpy as np
import pandas as pd

# Open the raster file
with rasterio.open(r"Y:\ATD\DEM_Alignment\East_Troublesome_Alignment\LIDAR_Alignment\LM2\LM2 070923 Veg Masked\LM2_070923_5cm_veg_masked_QGIS_ET_lower_LIDAR_2020_1m_DEM_reproj_nuth_x+1.39_y-0.92_z+0.84_align_diff.tif") as src:
    array = src.read(1)  # Read the first band
    affine = src.transform

# Generate coordinates
x, y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
x, y = x * affine[0] + affine[2], y * affine[4] + affine[5]

# Flatten the arrays
df = pd.DataFrame({
    'X': x.flatten(),
    'Y': y.flatten(),
    'Value': array.flatten()
})

# Save to CSV
df.to_csv('output.csv', index=False)
