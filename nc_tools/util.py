import xarray as xr


def extract_variable_name(dataset: xr.Dataset):
    variable_name = max(dataset.data_vars, key=lambda var: len(dataset[var].dims))
    return variable_name