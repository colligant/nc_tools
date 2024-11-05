from __future__ import annotations

import xarray as xr
import yaml

# land mask definition
land_mask = xr.open_dataset('/discover/nobackup/tcolliga/masked_drivers/masked/cru_ts4.07.1901.2022.tmp.dat.nc')['tmp'].isel(time=0).notnull()
