from astropy.table import Table
import numpy as np
import pandas as pd

def _logit(x):
    return np.log(x / (1 - x))


def gaussian(x, mu=0, sigma=1):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -1 * (x - mu) ** 2 / (2 * sigma ** 2)
    )


def calc_metrics(z_phot, z_spec, outlier_threshold):
    delta_z_norm = (z_phot - z_spec) / (1 + z_spec)
    
    # Normalized median absolute deviation
    sigma_nmad = 1.4826 * np.median(
        np.abs(delta_z_norm - np.median(delta_z_norm))
    )
    
    bias = np.mean(delta_z_norm[np.abs(delta_z_norm) < outlier_threshold])
    
    # Number of objects larger than outlier threshold
    n_outlier = np.sum(np.abs(delta_z_norm) > outlier_threshold)
    percent_outlier = n_outlier * 100.0 / len(z_spec)
    
    return sigma_nmad, bias, percent_outlier


def load_C3R2(path_C3R2):
    phot = Table.read(path_C3R2 / 'cosmos_c3r2_photometric__catalog_2021april23.fits').to_pandas()
    phot.columns= phot.columns.str.lower()
    phot['id'] = phot['id'].map("COSMOS-{}".format)
    phot = phot.reindex(columns=(['photz'] + list([a for a in phot.columns if a != 'photz']) ))
    phot.rename(columns={'photz': 'zphot'}, inplace=True)
    
    cols_spec = ['id', 'zspec', 'zspec_quality', ]
    spec = pd.read_csv(
        path_C3R2 / 'c3r2_targets_2019mar28.txt',
        skiprows=17,
        delim_whitespace=True,
        usecols=(0, 6, 7),
        names = cols_spec,
    )
    
    obs = pd.merge(spec, phot, on='id')
    
    return obs


def load_C3R2_train_test(path_C3R2, train_frac=0.8, seed=200):
    obs = load_C3R2(path_C3R2)

    bands = np.array(['u', 'g', 'r', 'i', 'z', 'y', 'j', 'h', 'ks'])

    # use i band as reference
    ind_i = np.where(np.array(bands) == 'i')[0][0]
    ind_not_i = np.where(np.array(bands) != 'i')[0]

    mags = obs.loc[:, bands].values
    ind_good = np.logical_and(
        (np.sum(mags > 40, axis=1) == 0),
        (np.sum(mags < 10, axis=1) == 0),
    )
    mags_good = mags[ind_good]


    idx = np.random.choice(
        a=[True, False],
        size=(ind_good.sum()),
        p=[train_frac, 1 - train_frac],
    )

    colors = (mags_good[:, ind_not_i].T - mags_good[:, ind_i]).T

    colors_train = colors[idx]
    z_train = obs.zspec.values[ind_good][idx]

    colors_test = colors[~idx]
    z_test = obs.zspec.values[ind_good][~idx]
    
    return colors_train, z_train, colors_test, z_test


def get_cell_coord(cell_coord, obj_cell_coord):
    return np.array([it == cell_coord for it in obj_cell_coord])


def objects_per_cell(som, data, n_som_xy):
    n_train = data.shape[0]
    cell_counts = np.zeros(n_som_xy)
    obj_cell = np.ones((n_train, 2), dtype=int) * -1
    obj_cell_coord = []

    for ii, x in enumerate(data):
        w = som.winner(x)
        cell_counts[w] += 1
        obj_cell[ii] = w
        obj_cell_coord.append(w)

    cell_counts[cell_counts == 0] = cell_counts[cell_counts == 0] * np.nan
    return obj_cell_coord, cell_counts


def mean_z_per_cell(obj_cell_coord, z, n_som_xy):
    z_mean_cell = np.zeros(n_som_xy) * np.nan
    for ii in range(n_som_xy[0]):
        for jj in range(n_som_xy[1]):
            ind = get_cell_coord((ii, jj), obj_cell_coord)
            if np.any(ind):
                z_mean_cell[ii, jj] = np.mean(z[ind])
    return z_mean_cell

def sigma_z_per_cell(obj_cell_coord, z, n_som_xy):
    z_sigma_cell = np.zeros(n_som_xy) * np.nan
    for ii in range(n_som_xy[0]):
        for jj in range(n_som_xy[1]):
            ind = get_cell_coord((ii, jj), obj_cell_coord)
            if np.any(ind):
                z_sigma_cell[ii, jj] = np.std(z[ind])
    return z_sigma_cell

def median_z_per_cell(obj_cell_coord, z, n_som_xy):
    z_median_cell = np.zeros(n_som_xy) * np.nan
    for ii in range(n_som_xy[0]):
        for jj in range(n_som_xy[1]):
            ind = get_cell_coord((ii, jj), obj_cell_coord)
            if np.any(ind):
                z_median_cell[ii, jj] = np.median(z[ind])
    return z_median_cell

def nmad_per_cell(obj_cell_coord, z, n_som_xy, z_true):
    nmad_cell = np.zeros(n_som_xy) * np.nan
    for ii in range(n_som_xy[0]):
        for jj in range(n_som_xy[1]):
            ind = get_cell_coord((ii, jj), obj_cell_coord)
            if np.any(ind):
                nmad_cell[ii, jj] = np.median(z[ind])
    return nmad_cell

def f_outlier_per_cell(obj_cell_coord, z, n_som_xy, z_true):
    f_outlier_cell = np.zeros(n_som_xy) * np.nan
    for ii in range(n_som_xy[0]):
        for jj in range(n_som_xy[1]):
            ind = get_cell_coord((ii, jj), obj_cell_coord)
            if np.any(ind):
                f_outlier_cell[ii, jj] = np.median(z[ind])
    return f_outlier_cell

def bias_per_cell(obj_cell_coord, z, n_som_xy, z_true):
    bias_cell = np.zeros(n_som_xy) * np.nan
    for ii in range(n_som_xy[0]):
        for jj in range(n_som_xy[1]):
            ind = get_cell_coord((ii, jj), obj_cell_coord)
            if np.any(ind):
                bias_cell[ii, jj] = np.median(z[ind])
    return bias_cell




