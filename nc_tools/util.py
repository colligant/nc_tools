from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import nc_tools.const as const


def extract_variable_name(dataset: xr.Dataset):
    return max(dataset.data_vars, key=lambda var: len(dataset[var].dims))


def gridarea_dataset_05deg(lat='latitude', lon='longitude') -> xr.Dataset:
    """
    Return gridarea for a 0.5x0.5deg grid for the input dataset.
    Tested on a different function that pulls nc_lat directly from an existing dataset.
    :param lon: name of lon array
    :param lat: name of lat array
    :return:
    """
    # 0.5 deg coordinates
    nc_lat = np.arange(-89.75, 90, step=0.5)
    nc_lon = np.arange(-179.75, 180, step=0.5)
    # cell area in m2
    global_cellarea = (
            111.13295 * 111.13295 * 0.5 * 0.5 * np.cos(np.radians(nc_lat)) * 1000 * 1000
    )
    global_cellarea = np.tile(global_cellarea, (len(nc_lon), 1)).T

    da = xr.DataArray(
        global_cellarea,
        dims=[lat, lon],
        coords={lat: nc_lat, lon: nc_lon},
        name='cell_area',
    )

    return xr.Dataset({'cell_area': da})


def weight_ds_land_mask(ds: xr.Dataset) -> xr.Dataset:
    """
    Returns a weighted dataset
    :param ds:
    :return:
    """
    landmask = const.land_mask.sel(
        lat=slice(
            min(
                ds.lat.values,
                max(ds.lat.values),
            ),
        ),
        lon=slice(
            min(ds.lon.values),
            max(ds.lon.values),
        ),
    )

    global_cellarea = gridarea_dataset_05deg().cell_area

    # Calculate land cell areas
    land_cellarea = global_cellarea * landmask
    land_cellarea = land_cellarea.to_dataset()

    land_cellarea = land_cellarea / land_cellarea.sum(skipna=True)
    weighted_data = ds[extract_variable_name(ds)].weighted(land_cellarea)
    return weighted_data


def land_mask_weighted_aggregation(ds: xr.Dataset, agg_type='sum') -> pd.DataFrame:
    """
    Use on climate rasters.
    :param ds:
    :param agg_type:
    :return:
    """
    # first weight by cell area after selecting the subregion

    weighted_data = weight_ds_land_mask(ds=ds)

    if agg_type == 'sum':
        weighted_sum = weighted_data.sum(dim=('lat', 'lon'), skipna=True)
        df = (
            weighted_sum.resample(time='Y')
            .sum()
            .compute()
            .to_dataframe()
        )
    if agg_type == 'mean':
        weighted_mean = weighted_data.mean(dim=('lat', 'lon'), skipna=True)
        df = (
            weighted_mean.resample(time='Y')
            .mean()
            .compute()
            .to_dataframe()
        )

    return df

def weight_by_cell_area(ds: xr.Dataset) -> pd.DataFrame:
    """
    Use on LPJ outputs to remove the m-2 unit.
    :param ds: Single variable dataset to apply weights to.
    :return:
    """
    # first weight by cell area after selecting the subregion
    variable = extract_variable_name(ds)
    area = gridarea_dataset_05deg()
    return ds[variable].weighted(area.cell_area)
