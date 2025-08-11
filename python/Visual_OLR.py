import netCDF4 as nc
import numpy as np

file_path = r"OLR.nc"
olr = nc.Dataset(file_path, mode="r").variables
data = olr['olra'][:, :, 36:109]
data = np.array(data)

print(data[:, 53:73, 53:73])