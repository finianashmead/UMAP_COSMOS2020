# umapz: functions for using UMAP (or SOM) dimensionality reduced 
# color space to interpolate redshifts, and to test the 
# quality of these mappings and interpolations

import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import pandas as pd
pd.set_option('display.max_columns', None)
from importlib import reload
from sklearn import neighbors
import utils

from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
import astropy.units as u

#animation imports
import os
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set(style='white', rc={'figure.figsize':(14, 12), 'animation.html': 'html5'})
import colorcet as cc
import subprocess
import glob

## specify what versions of each import numpy, pandas, plotly, etc. are used and work 
## (this can be done with pip requirements file)

## CROSSMATCH
def crossmatch(cat1, cat2, coords1, coords2, id, threshold=0.5):
    '''This function calculates the nearest neighbor in cat1 to each object in cat2, and the angular separation 
    between them in arcseconds

    Parameters
    ----------
    cat1: pandas DataFrame
        the catalog to be searched for matches to cat2 objects
    cat2: pandas DataFrame
        the catalog for which each object will be matched to one in cat1
    coords1: tuple of strings
        the column names of the coordinates in cat1, e.g. ('RA', 'DEC')
    coords2: tuple of strings
        the column names of the coordinates in cat2, e.g. ('RA', 'DEC')
    id: string
        the column name for the source IDs in cat1
    threshold: float, default=0.5
        angular separation threshold for a match between two objects, in arcseconds

    Returns
    ---------- 
    cat2: pandas DataFrame with added columns 'nearest_cat1_ID', 'nearest_cat1_sep'
    '''
    cat1 = cat1.copy()
    cat2 = cat2.copy()

    # Create SkyCoord objects for the first catalog (cat1)
    coords1 = SkyCoord(ra=cat1[coords1[0]].values * u.degree, dec=cat1[coords1[1]].values * u.degree)

    # Create a tree for efficient nearest neighbor search
    kdtree = cKDTree(np.column_stack((coords1.ra.deg, coords1.dec.deg)))

    # Create SkyCoord objects for the second catalog (cat2)
    coords2 = SkyCoord(ra=cat2[coords2[0]].values * u.degree, dec=cat2[coords2[1]].values * u.degree)

    # Query the nearest neighbors in cat1 for each point in cat2
    distances, nearest_indices = kdtree.query(np.column_stack((coords2.ra.deg, coords2.dec.deg)))

    # Convert distances from degrees to arcseconds
    distances_arcsec = distances * 3600  # 1 degree = 3600 arcseconds

    # Gather the results for nearest IDs and distances
    nearest_ids = cat1[id].iloc[nearest_indices].values
    nearest_distances = distances_arcsec

    # Add the results to cat2 DataFrame
    cat2['nearest_cat1_ID'] = nearest_ids
    cat2['nearest_cat1_sep'] = nearest_distances

    cat2 = cat2.copy()[cat2['nearest_cat1_sep']<threshold]

    matched1 = cat1[cat1[id].isin(cat2['nearest_cat1_ID'])]
    matched_merged = pd.merge(cat2, matched1, left_on='nearest_cat1_ID', right_on=id, how='left')

    return matched_merged

## PERFORMANCE METRICS
def calc_nmad(i_vals, t_vals):
    '''This function calculates the Normalized Median Absolute Deviation (NMAD) between 
    the UMAP-interpolated/SOM-assigned and truth redshift values for a given test sample

    Parameters
    ----------
    i_vals: pandas DataFrame column of floats
        the UMAP-interpolated / SOM-assigned redshift values
    t_vals: pandas DataFrame column of floats
        the "truth" redshift values (LePhare or spectroscopic) of the same sources

    Returns
    ---------- 
    float: NMAD
    '''
    delta_z = (i_vals - t_vals) / (1 + t_vals)
    med_delta_z = delta_z.median()
    
    median_absolute_deviation = np.median(np.abs(delta_z - med_delta_z))
    NMAD = 1.4826 * median_absolute_deviation    
    return NMAD

def calc_bias(i_vals, t_vals):
    '''This function calculates the bias between the UMAP-interpolated/SOM-assigned
    and truth redshift values for a given test sample, where the bias is the mean value of 
    the difference between the interpolated/assigned redshift and truth redshift, divided by
    the truth redshift plus one, i.e. delta-z/(1+z)

    Parameters
    ----------
    i_vals: pandas DataFrame column of floats
        the UMAP-interpolated / SOM-assigned redshift values
    t_vals: pandas DataFrame column of floats
        the "truth" redshift values (LePhare or spectroscopic) of the same sources
    threshold: float
        the value of delta-z/(1+z) above which a source will be considered an outlier

    Returns
    ---------- 
    float: bias
    '''
    bias_i = (i_vals - t_vals) / (1 + t_vals)
    bias = bias_i.mean()
    return bias

def f_outlier(i_vals, t_vals, missing=0, threshold=0.15):
    '''This function calculates the fraction of outliers between the UMAP-interpolated/SOM-assigned
     and truth redshift values for a given test sample, where a source is considered to be an
     outlier if the difference between the interpolated/assigned redshift and truth redshift, divided
     the truth redshift plus one, i.e. delta-z/(1+z), is greater than the threshold

    Parameters
    ----------
    i_vals: pandas DataFrame column of floats
        the UMAP-interpolated / SOM-assigned redshift values
    t_vals: pandas DataFrame column of floats
        the "truth" redshift values (LePhare or spectroscopic) of the same sources
    missing: int, optional
        an integer number of missing sources (i.e. sources in SOM cells with no training/labeled
        redshifts) to be added to the count of outliers
    threshold: float
        the value of delta-z/(1+z) above which a source will be considered an outlier

    Returns
    ---------- 
    float: fraction of outliers
    '''
    delta_z_norm = (i_vals - t_vals) / (1 + t_vals)
    # Number of objects larger than outlier threshold
    n_outlier = np.sum(np.abs(delta_z_norm) > threshold)
    n_outlier+=missing
    percent_outlier_ = n_outlier / (missing+len(t_vals))
    return percent_outlier_

## INTERPOLATION
def interpolator(df1, df2, map='UMAP3D', dim=3, nn=15, calc='median', z_true='lp_photoz'):    
    '''This function assigns redshifts to the sources in df2 based on the sources in df1.
    
    In the case of SOM maps, this can be the mean or median of the redshifts of df1 sources
    in the same cell. In the case of UMAP maps, this can be the mean, inverse-distance-weighted
    mean, or median of the nn nearest neighbors in the UMAP embedding.

    Parameters
    ----------
    df1: pandas DataFrame of floats
        DataFrame containing the labeled/training data
    df2: pandas DataFrame of floats 
        DataFrame containing the unlabeled/test data
    map: string
        the labeling of the spatial coordinates, can be any combination of UMAP/densMAP 2D/3D, 
        or SOM, i.e. UMAP3D, densMAP2D, SOM, etc.
    dim: int
        the dimensionality of the space (2 and 3 are supported)
    nn: int
        the number of neighbors to use in the interpolation (UMAP case), or the length of the shorter side
        of the SOM grid (1:2 aspect ratio)
    calc: string 
        can be 'mean', inverse-distance-weighted mean ('inverse'), or 'median' (only 'mean' and 'median' supported for SOM)

    Returns
    ----------
    DataFrame: copy of df2 modified to include a TEST_Z column containing the redshift values interpolated based on df1
    '''
    n_som_xy = (nn, 2*nn)
    coords = []
    for i in range(dim): 
        coords.append(map + '-' + str(i+1))
    
    # Create a dataframe with only coordinates, redshift, and sSFR from df1
    df_combined = df1[coords+[z_true, 'lp_sSFR_best']].copy()
    df2_ = df2.copy()
    
    # Separate known values of z and corresponding values of the coordinates
    known_z = df_combined[z_true]
    known_map = df_combined[coords]
    
    # Fit a nearest neighbors model on the known coordinates
    nn_model = neighbors.NearestNeighbors(n_neighbors=nn)
    nn_model.fit(known_map)
    
    # Separate unknown values of z (for interpolation) and corresponding coordinates
    # Find the distances and indices of the nearest neighbors for each point in 'unknown_map'
    unknown_map = df2_[coords]
    distances, indices = nn_model.kneighbors(unknown_map)
    
    if calc == 'inverse':
        # Calculate weights based on distances (inverse distance weighting)
        weights = 1.0 / distances
        # Normalize the weights
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        # Compute the weighted average of z values of the nearest neighbors
        interpolated_z = np.sum(weights * known_z.values[indices], axis=1)
    elif calc == 'mean':
        #if map == 'SOM':
        if 'SOM' in map:
            z_train = known_z.values          
            occ_train = known_map[coords].values
            occ_train_list = [tuple(row) for row in occ_train]
            z_mean_cell_train = utils.mean_z_per_cell(occ_train_list, z_train, n_som_xy)
            interpolated_z = z_mean_cell_train[unknown_map[coords[0]], unknown_map[coords[1]]]
        else:
            interpolated_z = np.mean(known_z.values[indices], axis=1)
    elif calc == 'median':
        #if map == 'SOM':
        if 'SOM' in map:
            z_train = known_z.values  
            occ_train = known_map[coords].values
            occ_train_list = [tuple(row) for row in occ_train]
            z_median_cell_train = utils.median_z_per_cell(occ_train_list, z_train, n_som_xy)
            interpolated_z = z_median_cell_train[unknown_map[coords[0]], unknown_map[coords[1]]]            
        else:
            interpolated_z = np.median(known_z.values[indices], axis=1)
    
    # Add the interpolated z values to df2
    df2_['TEST_Z'] = interpolated_z
    df2_ = df2_.dropna(subset=['TEST_Z'])
    
    return df2_

def binned_stats(fullcat, testcat, iv, tv, bin_var, bin_min, bin_max, bin_size):
    '''Function (used as helper function in z_binned_stats and z_binned_umap_vs_colors) that 
    calculates the bin centers as well as the number of objects (counts), fraction of outliers, 
    NMAD, bias, and fraction of outliers (trained cells only) for each bin

    Parameters
    ----------
    fullcat: pandas DataFrame
        DataFrame containing the full test/target set catalog
    testcat: pandas DataFrame
        DataFrame containing the catalog of objects with UMAP-kNN-z/SOM-z estimates,
        i.e. the output of the interpolator function
    iv: string
        string defining the column of testcat containing the interpolated redshift values,
        i.e. 'TEST_Z' in the standard output of the interpolator function
    tv: string
        string defining the column of testcat containing the "true" redshift values,
        in our standard application this is the LePHARE redshifts: 'lp_photoz'
    bin_var: string
        string defining the column according to which the data should be binned, 
        in the z_binned_stats and z_binned_umap_vs_colors functions the LePHARE 
        redshift is used, but other variables, e.g., photometric magnitudes, 
        could be used as well
    bin_min: float
        the lower limit of the binning
    bin_max: float
        the upper limit of the binning
    bin_size: float
        the width of the bins 
    
    Returns
    ----------
    bin_centers: list
        list of bin centers
    counts: list
        list of the number of objects in each bin
    outliers:
        list of the fraction of outliers in each bin
    nmads:
        list of the NMAD values in each bin
    biases:
        list of the bias values in each bin
    nn_outliers:
        list of the fraction of outliers in each bin (trained cells only)
    '''
    counts = []
    outliers = []
    nn_outliers = []
    nmads = []
    biases = []

    bin_centers = np.arange(bin_min+bin_size/2, bin_max+bin_size/2, bin_size)
    for i, bc in enumerate(bin_centers):
        low_end = bc - bin_size/2
        high_end = bc + bin_size/2
        testcat_i = testcat[(testcat[bin_var]>=low_end) & (testcat[bin_var]<high_end)]
        fullcat_i = fullcat[(fullcat[bin_var]>=low_end) & (fullcat[bin_var]<high_end)]

        counts.append(len(testcat_i))
        outliers.append(f_outlier(i_vals=testcat_i[iv], t_vals=testcat_i[tv], missing=len(fullcat_i)-len(testcat_i), threshold=0.15))
        nn_outliers.append(f_outlier(i_vals=testcat_i[iv], t_vals=testcat_i[tv], missing=0., threshold=0.15))
        nmads.append(calc_nmad(i_vals=testcat_i[iv], t_vals=testcat_i[tv]))
        biases.append(calc_bias(i_vals=testcat_i[iv], t_vals=testcat_i[tv]))

    return bin_centers, counts, outliers, nmads, biases, nn_outliers

def binned_train(traincat, bin_var, bin_min, bin_max, bin_size):
    '''This function calculates the bin centers and number of training objects in each bin
    for a desired binning scheme

    Parameters
    ----------
    traincat: pandas DataFrame
        DataFrame containing the training dataset
    bin_var: string
        string defining the column according to which the data should be binned, 
        in the z_binned_stats and z_binned_umap_vs_colors functions the LePHARE 
        redshift is used, but other variables, e.g., photometric magnitudes, 
        could be used as well
    bin_min: float
        the lower limit of the binning
    bin_max: float
        the upper limit of the binning
    bin_size: float
        the width of the bins 

    Returns
    ----------
    bin_centers: list
        list of the centers of each bin
    train_counts: list
        list of the number of training objects in each bin
    '''
    train_counts = []

    bin_centers = np.arange(bin_min+bin_size/2, bin_max+bin_size/2, bin_size)
    for i, bc in enumerate(bin_centers):
        low_end = bc - bin_size/2
        high_end = bc + bin_size/2
        traincat_i = traincat[(traincat[bin_var]>=low_end) & (traincat[bin_var]<high_end)]

        train_counts.append(len(traincat_i))

    return bin_centers, train_counts

def interpolator_colors(df1, df2, coords, nn=15, calc='median', z_true='lp_photoz', p=2):    
    '''This function assigns redshifts to the sources in df2 based on the sources in df1 neighboring 
    them in the color space.
    
    This can be the mean, inverse-distance-weighted mean, or median of the nn nearest neighbors in 
    the color space.

    Parameters
    ----------
    df1: pandas DataFrame of floats
        DataFrame containing the labeled/training data
    df2: pandas DataFrame of floats 
        DataFrame containing the unlabeled/test data
    map: string
        the labeling of the spatial coordinates, can be any combination of UMAP/densMAP 2D/3D, 
        or SOM, i.e. UMAP3D, densMAP2D, SOM, etc.
    dim: int
        the dimensionality of the space (2 and 3 are supported)
    nn: int
        the number of neighbors to use in the interpolation (UMAP case), or the length of the shorter side
        of the SOM grid (1:2 aspect ratio)
    calc: string 
        can be 'mean', inverse-distance-weighted mean ('inverse'), or 'median' (only 'mean' and 'median' supported for SOM)
    z_true: string
        redshift to use for the objects in df1 that redshift estimation is based on, can be 'lp_photoz' or 'specz'
    p: integer
        "p" input to neighbors.NearestNeighbors; p=1 for Manhattan distance, p=2 for Euclidean
        
    Returns
    ----------
    DataFrame: copy of df2 modified to include a TEST_Z column containing the redshift values interpolated based on df1
    '''
    # Create a dataframe with only coordinates, redshift, and sSFR from df1
    df_combined = df1[coords+[z_true, 'lp_sSFR_best']].copy()
    df2_ = df2.copy()
    
    # Separate known values of z and corresponding values of the coordinates
    known_z = df_combined[z_true]
    known_map = df_combined[coords]
    
    # Fit a nearest neighbors model on the known coordinates
    nn_model = neighbors.NearestNeighbors(n_neighbors=nn, p=p)
    nn_model.fit(known_map)
    
    # Separate unknown values of z (for interpolation) and corresponding coordinates
    # Find the distances and indices of the nearest neighbors for each point in 'unknown_map'
    unknown_map = df2_[coords]
    distances, indices = nn_model.kneighbors(unknown_map)
    
    if calc == 'inverse':
        # Calculate weights based on distances (inverse distance weighting)
        weights = 1.0 / distances
        # Normalize the weights
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        # Compute the weighted average of z values of the nearest neighbors
        interpolated_z = np.sum(weights * known_z.values[indices], axis=1)
    elif calc == 'mean':
        interpolated_z = np.mean(known_z.values[indices], axis=1)
    elif calc == 'median':
        interpolated_z = np.median(known_z.values[indices], axis=1)
    
    # Add the interpolated z values to df2
    df2_['TEST_Z'] = interpolated_z
    df2_ = df2_.dropna(subset=['TEST_Z'])
    
    return df2_

def plot_som(phot_df, spec_df, map, size, filename=None):
    '''
    This function generates the 3x3 SOM plot (corresponding to Fig. 2 in the paper)

    Parameters
    -------------------
    phot_df: DataFrame containing full photometric catalog, must contain some form of 'SOM-1' and 'SOM-2' coords
    spec_df: DataFrame containing spec-z catalog, must contain some form of 'SOM-1' and 'SOM-2' coords
    map: SOM lead string, e.g., SOM, SOM8, SOM75, etc.
    size: tuple of integers describing SOM size

    Returns
    ------------------
    3x3 SOM plot
    counts_all, counts_lp, counts_s
    '''
    df1 = phot_df.copy()
    df2 = spec_df.copy()
    n_som_xy = size
    coords = [map+'-1', map+'-2']

    # 1. Count of entries at each (X, Y) grid point
    count_df_all = df1.groupby(coords).size().reset_index(name='count')
    cell_counts_50 = count_df_all.pivot(index=coords[0], columns=coords[1], values='count').fillna(0).to_numpy()

    # 2. Median of 'z' at each (X, Y)
    z_median_df = df1.groupby(coords)['lp_photoz'].median().reset_index()
    z_median_cell_50 = z_median_df.pivot(index=coords[0], columns=coords[1], values='lp_photoz').to_numpy()

    # 3. Median of 's' at each (X, Y)
    s_median_df = df1.groupby(coords)['lp_sSFR_best'].median().reset_index()
    ssfr_median_cell_50 = s_median_df.pivot(index=coords[0], columns=coords[1], values='lp_sSFR_best').to_numpy()

    # 1. Count of entries at each (X, Y) grid point
    count_df_s = df2.groupby(coords).size().reset_index(name='count')
    cell_counts_50_s = count_df_s.pivot(index=coords[0], columns=coords[1], values='count').fillna(0).to_numpy()

    # 2. Median of 'z' at each (X, Y)
    z_median_df = df2.groupby(coords)['specz'].median().reset_index()
    z_median_cell_50_s = z_median_df.pivot(index=coords[0], columns=coords[1], values='specz').to_numpy()

    # 3. Median of 's' at each (X, Y)
    s_median_df = df2.groupby(coords)['lp_sSFR_best'].median().reset_index()
    ssfr_median_cell_50_s= s_median_df.pivot(index=coords[0], columns=coords[1], values='lp_sSFR_best').to_numpy()

    df_random = df1.sample(n=len(df1), random_state=2)
    train_size = len(df2)
    df_test_lp = df_random[:len(df1)-train_size]
    df_train_lp = df_random[len(df1)-train_size:len(df1)]

    # 1. Count of entries at each (X, Y) grid point
    count_df_lp = df_train_lp.groupby(coords).size().reset_index(name='count')
    cell_counts_50_lp = count_df_lp.pivot(index=coords[0], columns=coords[1], values='count').fillna(0).to_numpy()

    # 2. Median of 'z' at each (X, Y)
    z_median_df = df_train_lp.groupby(coords)['lp_photoz'].median().reset_index()
    z_median_cell_50_lp = z_median_df.pivot(index=coords[0], columns=coords[1], values='lp_photoz').to_numpy()

    # 3. Median of 's' at each (X, Y)
    s_median_df = df_train_lp.groupby(coords)['lp_sSFR_best'].median().reset_index()
    ssfr_median_cell_50_lp = s_median_df.pivot(index=coords[0], columns=coords[1], values='lp_sSFR_best').to_numpy()

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.cm as cm

    data_sets = [
        (np.log10(cell_counts_50), z_median_cell_50, ssfr_median_cell_50),
        (np.log10(cell_counts_50_lp), z_median_cell_50_lp, ssfr_median_cell_50_lp),
        (np.log10(cell_counts_50_s), z_median_cell_50_s, ssfr_median_cell_50_s)
    ]

    labels = (r'$N$', r'$z$', r'sSFR [yr$^{-1}$]')
    vmaxs = (2, 5, -8)
    vmins = (0, 0, -12)
    fsz = 28

    # Color maps
    mycm1 = plt.cm.get_cmap('viridis').copy()
    mycm1.set_bad('gray', 0.2)
    mycm2 = plt.cm.get_cmap('cet_CET_R1').copy()
    mycm2.set_bad('gray', 0.2)
    mycm3 = plt.cm.get_cmap('cet_CET_D1A_r').copy()
    mycm3.set_bad('gray', 0.2)
    cmaps = (mycm1, mycm2, mycm3)

    # Layout
    n_rows, n_cols = 3, 3
    width, height = 0.25, 0.25
    x_spacing, y_spacing = 0.05, 0.015
    lefts = [0.05 + i * (width + x_spacing) for i in range(n_cols)]
    bottoms = [0.05 + (2 - i) * (height + y_spacing) for i in range(n_rows)]  # top to bottom

    fig = plt.figure(figsize=(10, 20))
    axes_grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    # Loop over data rows and columns
    for row in range(n_rows):
        data = data_sets[row]
        for col in range(n_cols):
            left = lefts[col]
            bottom = bottoms[row]
            ax = fig.add_axes([left, bottom, width, height])
            axes_grid[row][col] = ax

            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            extent = (0.5, n_som_xy[0] + 0.5, 0.5, n_som_xy[1] + 0.5)
            im = ax.imshow(data[col].T, extent=extent, origin='lower',
                        cmap=cmaps[col], vmin=vmins[col], vmax=vmaxs[col])

            if row == 0:
                # Create separate, fixed-position colorbar axes above each column
                cbar_height = 0.02
                cbar_bottom = bottom + height + 0.01
                cbar_ax = fig.add_axes([left, cbar_bottom, width, cbar_height])
                cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                cb.set_label(labels[col], fontsize=fsz)
                cb.ax.xaxis.set_ticks_position('top')
                cb.ax.xaxis.set_label_position('top')

                # Tick logic (same as before)
                if col == 0:
                    ticklabels = np.array([1, 10, 30, 100])
                    ticks_loc = np.log10(ticklabels)
                elif col == 2:
                    ticks_loc = np.array([-12, -11, -10, -9, -8])
                    ticklabels = ticks_loc
                else:
                    ticks_loc = np.array([0, 1, 2, 3, 4, 5])
                    ticklabels = ticks_loc
                cb.set_ticks(ticks_loc)
                cb.ax.set_xticklabels(ticklabels, rotation=45, fontsize=int(0.75*fsz))

    # add vertical labels here!!
    label_x = 0.006
    fig.text(label_x, 0.45, 'random sample', va='center', ha='left',
                rotation=90, fontsize=fsz)
    fig.text(label_x, 0.18, 'spec-'+r'$z$'+' CL>95', va='center', ha='left',
                rotation=90, fontsize=fsz)
    
    #plt.show()

    if filename!=None:
        fig.savefig(filename, dpi=500, bbox_inches='tight')

    return count_df_all, count_df_lp, count_df_s

## PLOTS FROM THE PAPER
def plot_distributions(phot_df, spec_df, filename=None):
    '''
    This function generates the 4-panel plot of the redshift, i-band magnitude, g-z color, and z-y color (1.4<z<1.5)
    distributions of the photometric and spectroscopic samples (corresponding to Figure 1 of the paper)

    Parameters
    -------------------
    phot_df: DataFrame 
        DataFrame containing full photometric catalog
    spec_df: DataFrame 
        DataFrame containing spec-z catalog
    filename: string, default=None 
        if not None, the image will be written out with the specified filename,
        must include format tail, ex: 'figure1.png'

    Returns
    ------------------
    2x2 distributions plot, optionally saved to directory as 'filename'
    '''
    # defining some things for the figure
    phot_df['g-z'] = phot_df['HSC_g_MAG'] - phot_df['HSC_z_MAG']
    spec_df['g-z'] = spec_df['HSC_g_MAG'] - spec_df['HSC_z_MAG']
    high_z = 1.5
    low_z = 1.4
    photo_zcut = phot_df.copy()[(phot_df['lp_photoz']<high_z) & (phot_df['lp_photoz']>low_z)]
    spec_zcut = spec_df.copy()[(spec_df['specz']<high_z) & (spec_df['specz']>low_z)]
    color = 'z-y'

    fsz = 20
    lbsz = 15
    fig, axs = plt.subplots(2,2, figsize=(20,12))
    axs[0,0].hist(phot_df['lp_photoz'], bins=np.linspace(0,4,121), alpha=0.6, density=True, color='tab:orange', label='photometric sample');
    axs[0,0].hist(spec_df['specz'], bins=np.linspace(0,4,121), alpha=1., histtype='step', density=True, lw=1.5, label='spectroscopic sample');
    axs[0,1].hist(phot_df['g-z'], bins=np.linspace(-0.5,5,121), alpha=0.6, density=True, color='tab:orange')#, label='photo-z');
    axs[0,1].hist(spec_df['g-z'], bins=np.linspace(-0.5,5,121), alpha=1., histtype='step', density=True, lw=1.5)#, label='spec-z');
    axs[1,0].hist(phot_df['HSC_i_MAG'], bins=np.linspace(18,24.5,121), alpha=0.6, density=True, color='tab:orange')#, label='photo-z');
    axs[1,0].hist(spec_df['HSC_i_MAG'], bins=np.linspace(18,24.5,121), alpha=1., histtype='step', density=True, lw=1.5)#, label='spec-z');
    axs[0,0].set_xlabel(r'redshift', fontsize=fsz)
    axs[0,1].set_xlabel(r'HSC $g-z$ (mag)', fontsize=fsz)
    axs[1,0].set_xlabel(r'HSC $i$ (mag)', fontsize=fsz)
    axs[0,0].set_ylabel(r'normalized density', fontsize=fsz)
    axs[0,1].set_ylabel(r'normalized density', fontsize=fsz)
    axs[1,0].set_ylabel(r'normalized density', fontsize=fsz)
    axs[0,0].tick_params(axis='x', labelsize=lbsz)
    axs[0,0].tick_params(axis='y', labelsize=lbsz)
    axs[0,1].tick_params(axis='x', labelsize=lbsz)
    axs[0,1].tick_params(axis='y', labelsize=lbsz)
    axs[1,0].tick_params(axis='x', labelsize=lbsz)
    axs[1,0].tick_params(axis='y', labelsize=lbsz)
    axs[1,1].tick_params(axis='x', labelsize=lbsz)
    axs[1,1].tick_params(axis='y', labelsize=lbsz)

    axs[1,1].hist(photo_zcut[color], bins=np.linspace(-0.5,1.,51), alpha=0.6, density=True, color='tab:orange', label='photometric sample');
    axs[1,1].hist(spec_zcut[color], bins=np.linspace(-0.5,1.,51), alpha=1., histtype='step', density=True, lw=1.5, label='spectroscopic sample');
    axs[1,1].set_xlabel(r'HSC $z-y$ (mag)', fontsize=fsz)
    axs[1,1].set_ylabel(r'normalized density', fontsize=fsz)
    axs[1,1].text(-0.5, 2.6, r'$1.4 < z < 1.5$', fontsize=25)

    axs[0,0].legend(fontsize=fsz);
    # save figure as desired
    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')

# Function to rotate plot
def rotate_plot(ax, angle):
    '''
    helper function for save_frame that rotates the 3d plot about the vertical axis
    '''
    ax.view_init(elev=10, azim=angle)

# Function to plot data (plot_data3 in animation_simplified2)
def plot_data(df, color='lp_photoz', cbar=True):
    '''Helper function used in save_frame to create 3d axes with the appropriate data plotted,
    and an optional color bar

    Parameters
    ----------
    df: pandas DataFrame
        DataFrame containing the data to be plotted
    color: string, default='lp_photoz'
        string specifying the color-coding for the plot, 'lp_photoz', 'Match_specz', 'specz',
        'lp_sSFR_best', and 'black' are supported
    cbar: bool, default=True
        boolean defining whether the figure should include a color bar 

    Returns
    ----------
    matplotlib Figure and Axes objects, used in save_frame
    '''

    fig = plt.figure(figsize=(10, 8), dpi=100)  # Increase overall figure size
    ax = fig.add_axes([0.0, 0.0, 0.95, 1.0], projection='3d') 
    fig.subplots_adjust(bottom=0, top=1)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.margins(y=0)

    # Create scatter plot
    if color == 'lp_photoz':
        sc = ax.scatter(df['Z1'], df['X1'], -df['Y1'], c=df[color], s=1., alpha=0.7, cmap=get_cmap('cet_CET_R1'), vmin=0, vmax=5)
    elif color == 'Match_specz':
        sc = ax.scatter(df['Z1'], df['X1'], -df['Y1'], c=df[color], s=1, alpha=0.7, cmap=get_cmap('cet_CET_R1'), vmin=0, vmax=5)
    elif color == 'specz':
        sc = ax.scatter(df['Z1'], df['X1'], -df['Y1'], c=df[color], s=1, alpha=0.7, cmap=get_cmap('cet_CET_R1'), vmin=0, vmax=5)
    elif color == 'lp_sSFR_best':
        sc = ax.scatter(df['Z1'], df['X1'], -df['Y1'], c=df[color], s=1., alpha=0.7, cmap=get_cmap('cet_CET_D1A_r'), vmin=-12, vmax=-8)
    elif color == 'black':
        sc = ax.scatter(df['Z1'], df['X1'], -df['Y1'], c='k', s=7, alpha=1.)
    else:
        sc = ax.scatter(df['Z1'], df['X1'], -df['Y1'], c=df[color], s=1, alpha=0.7, cmap=get_cmap('cet_CET_R1'))

    # Add colorbar with manual size control
    if cbar:
        # Custom position: [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Narrow and tall colorbar
        cbar = fig.colorbar(sc, cax=cbar_ax)

        # Label settings
        if color == 'lp_photoz':
            cbar.set_label(r'$z_\mathrm{LePHARE}$', fontsize=25)
        elif color == 'Match_specz':
            cbar.set_label(r'$z_\mathrm{spec}$', fontsize=25)
        elif color == 'specz':
            cbar.set_label(r'$z_\mathrm{spec}$', fontsize=25)
        elif color == 'lp_sSFR_best':
            cbar.set_label(r'sSFR [yr$^{-1}$]', fontsize=25)

        cbar.ax.tick_params(labelsize=20)
        

    return fig, ax

# combine and animate the frames into an mp4 video
def combine_and_animate(z_phot, ssfr_phot, z_spec, ssfr_spec, framerate=15):
    '''Combine image sequences from four directories, corresponding to the photometric and spectroscopic
    datasets color-coded by redshift and sSFR, into a four-panel mp4 animation.

    Parameters
    ----------
    z_phot: string
        Path to directory containing z_phot images (frame_%04d.png)
    ssfr_phot: string 
        Path to directory containing ssfr_phot images
    z_spec: string
        Path to directory containing z_spec images
    ssfr_spec: string 
        Path to directory containing ssfr_spec images
    framerate: int
        Frame rate for the final mp4 animation (default: 15)
    
    Returns
    ----------
    mp4 video of the rotating four-panel plot of the UMAP embedding
    '''

    # --- 1. Create combined two-panel (phot) ---
    twopanel_phot = "twopanel_phot"
    os.makedirs(twopanel_phot, exist_ok=True)
    print(f"Creating {twopanel_phot}...")

    # Create a temporary padded z_phot directory
    z_phot_padded = "z_phot_padded"
    os.makedirs(z_phot_padded, exist_ok=True)

    # Pad z_phot images by 1 pixel on each side
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", os.path.join(z_phot, "frame_%04d.png"),
        "-vf", "pad=iw+2:ih:1:0:color=white",
        os.path.join(z_phot_padded, "frame_%04d.png")
    ], check=True)

    # Combine the padded z_phot with ssfr_phot
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", os.path.join(z_phot_padded, "frame_%04d.png"),
        "-i", os.path.join(ssfr_phot, "frame_%04d.png"),
        "-filter_complex", "[0][1]hstack=inputs=2",
        os.path.join(twopanel_phot, "frame_%04d.png")
    ], check=True)

    # --- 2. Create combined two-panel (spec) ---
    twopanel_spec = "twopanel_spec"
    os.makedirs(twopanel_spec, exist_ok=True)
    print(f"Creating {twopanel_spec}...")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", os.path.join(z_spec, "frame_%04d.png"),
        "-i", os.path.join(ssfr_spec, "frame_%04d.png"),
        "-filter_complex", "[0][1]hstack=inputs=2",
        os.path.join(twopanel_spec, "frame_%04d.png")
    ], check=True)

    # --- 3. Combine both two-panel sequences vertically ---
    fourpanel = "fourpanel"
    os.makedirs(fourpanel, exist_ok=True)
    print(f"Creating {fourpanel}...")

    subprocess.run([
        "ffmpeg",
        "-y",  # overwrite output files without asking
        "-i", os.path.join(twopanel_phot, "frame_%04d.png"),
        "-i", os.path.join(twopanel_spec, "frame_%04d.png"),
        "-filter_complex", "[0][1]vstack=inputs=2",
        os.path.join(fourpanel, "frame_%04d.png")
    ], check=True)

    # --- 4. Create animation ---
    print("Creating final four-panel animation (MP4)...")
    output_mp4 = os.path.join(fourpanel, "fourpanel_animation.mp4")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-framerate", str(framerate),
        "-i", os.path.join(fourpanel, "frame_%04d.png"),
        "-vf", "scale=iw-mod(iw\\,2):ih-mod(ih\\,2)",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_mp4
    ], check=True)

    print("Animation created successfully: {output_mp4}")

# save frame of 3d UMAP plot
def save_frame(num_frames, index, df, output_dir, title=None, cbar=True, color='lp_photoz', df2=None, color2=None):
    '''This function saves a frame of the 3d UMAP plot, combine_and_animate stitches these frames together
    to generate the mp4 animation of the rotating plot

    Parameters
    ----------
    num_frames: int
        the number of frames the desired animation will contain, this is used
        to define the rotation of the plot such that a 360° rotation will be completed
        in the full animation produced by combine_and_animate
    index: int
        the current frame (will be an integer between 0 and num_frames), used
        to define the rotation angle of the plot for each image
    df: pandas DataFrame
        DataFrame containing the dataset that will be plotted
    output_dir: string
        string defining the output directory to which the images will be written out
    title: string, default=None
        optional title to appear above the axes if not None
    cbar: bool, default=True
        boolean defining whether a color bar should appear to the right of the axes
    color: string, default='lp_photoz'
        the color-coding for the objects contained in df, 
    df2: optional pandas DataFrame, default=None
        second DataFrame to plot on the same axes, can be used to, e.g., visualize where
        in the UMAP embedding a subclass of objects are located (a particular magnitude bin, 
        AGN, some spatial localization, etc.)
    color2: optional string, default=None
        second color for the objects contained in df2, supported colors can be seen by expanding plot_data
    
    Returns
    ----------
    none, png images saved to directory specified by output_dir
    '''
    df = df.rename(columns={'UMAP3D-1':'X1', 'UMAP3D-2':'Y1', 'UMAP3D-3':'Z1'})
    
    fig = plt.figure(figsize=(10,8))#, dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    if index < num_frames:
        # Plot data and rotate once
        sc = plot_data(df, color, cbar=cbar)
        if color2!=None:
            plot_data(df2, color2, cbar=cbar)
        
        rotate_plot(sc[1], (index * 360 / num_frames)-95)

    if title!=None:
        ax.set_title(title, fontsize=16)
    
    filename = os.path.join(output_dir, f"frame_{index:04d}.png")
    print(f"Saving frame {index} to {filename}")
    plt.margins(y=0)
    plt.savefig(filename, bbox_inches='tight')#, pad_inches=0)#, dpi=200)
    plt.close('all')

    crop_filter = "crop=in_w:in_h-100:0:50"
    temp_filename = filename + ".tmp.png"  # temporary cropped version

    # Run ffmpeg and write to a temp file
    subprocess.run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", filename,
        "-vf", crop_filter,
        "-y", temp_filename
    ])

    # Replace the original with the cropped version
    os.replace(temp_filename, filename)
    print(f"Cropped and overwritten: {filename}")

# create a Figure with performance statistics binned by redshift, corresponding to Figure 5 in our paper
def z_binned_stats(fullcat_lp, fullcat_spec, umap_test_lp, umap_test_spec, som_test_lp, som_test_spec, i_vals='TEST_Z', t_vals='lp_photoz', bin_min=0.0, bin_max=3.0, bin_size=0.2, filename=None):
    '''This function creates a multi-panel plot of the performance statistics binned by redshift, 
    corresponding to Figure 5 in our paper.

    Parameters
    ----------
    fullcat_lp: pandas DataFrame
        DataFrame containing the full catalog of test objects, LePHARE-trained case
    fullcat_spec: pandas DataFrame
        DataFrame containing the full catalog of test objects, spec-z-trained case
    umap_test_lp: pandas DataFrame
        DataFrame containing the catalog of objects with UMAP-kNN-z redshift estimates, 
        LePHARE-trained case
    umap_test_spec: pandas DataFrame
        DataFrame containing the catalog of objects with UMAP-kNN-z redshift estimates, 
        spec-z-trained case
    som_test_lp: pandas DataFrame
        DataFrame containing the catalog of objects with SOM-z redshift estimates, 
        LePHARE-trained case
    som_test_spec: pandas DataFrame
        DataFrame containing the catalog of objects with SOM-z redshift estimates, 
        spec-z-trained case
    i_vals: string, default='TEST_Z'
        string corresponding to the label of the DataFrame column containing the 
        redshift estimates interpolated using either UMAP-kNN-z or SOM-z
    t_vals: string, default='lp_photoz'
        string corresponding to the label of the DataFrame column containing the 
        "true" redshifts
    bin_min: float, default=0.0
        the lower limit of the redshift binning, and the lower x-axis limit for plotting
    bin_max: float, default=3.0
        the upper limit of the redshift binning
    bin_size: float, default=0.2
        the width of the redshift bins
    filename: string, default=None
        optional filename, if !=None the plot will be saved according to the format in the 
        string, e.g. 'z_binned_figure.png'

    Returns
    ----------
    figure: the plot!
    optionally saved to directory if filename!=None
    '''
    # calculate z-binned stats, LePHARE-trained case
    bin_centers, umap_counts, umap_outliers, umap_nmads, umap_biases, umap_nn_out = binned_stats(fullcat=fullcat_lp, testcat=umap_test_lp, 
                                                                         iv=i_vals, tv=t_vals, bin_var=t_vals, bin_min=bin_min, 
                                                                         bin_max=bin_max, bin_size=bin_size)
    bin_centers, som_counts, som_outliers, som_nmads, som_biases, som_nn_out = binned_stats(fullcat=fullcat_lp, testcat=som_test_lp, 
                                                                            iv=i_vals, tv=t_vals, bin_var=t_vals, bin_min=bin_min, 
                                                                            bin_max=bin_max, bin_size=bin_size)
    # calculate z-binned stats, spec-z-trained case
    bin_centers_s, umap_counts_s, umap_outliers_s, umap_nmads_s, umap_biases_s, umap_nn_out_s = binned_stats(fullcat=fullcat_spec, testcat=umap_test_spec, 
                                                                         iv=i_vals, tv=t_vals, bin_var=t_vals, bin_min=bin_min, 
                                                                         bin_max=bin_max, bin_size=bin_size)
    bin_centers_s, som_counts_s, som_outliers_s, som_nmads_s, som_biases_s, som_nn_out_s = binned_stats(fullcat=fullcat_spec, testcat=som_test_spec, 
                                                                            iv=i_vals, tv=t_vals, bin_var=t_vals, bin_min=bin_min, 
                                                                            bin_max=bin_max, bin_size=bin_size)
    # create the figure
    fig, axs = plt.subplots(3, 2, figsize=(16,12))

    fout_ymax = 1.05*np.max([np.max(som_outliers), np.max(umap_outliers), np.max(som_outliers_s), np.max(umap_outliers_s)])
    nmad_ymax = 1.05*np.max([np.max(som_nmads), np.max(umap_nmads), np.max(som_nmads_s), np.max(umap_nmads_s)])
    bias_ymax = 1.05*np.max([np.max(som_biases), np.max(umap_biases), np.max(som_biases_s), np.max(umap_biases_s)])

    # UPPER LEFT
    axs[0,0].plot(bin_centers, som_outliers, c='0.3', label='SOM (LePHARE-trained, all cells)', ls='--')
    axs[0,0].plot(bin_centers, som_nn_out, c='0.4', alpha=0.7, label='SOM (LePHARE-trained, cells with training redshifts)', ls='--')
    axs[0,0].fill_between(bin_centers, som_outliers, y2=som_nn_out, alpha=0.15, color='0.4')
    axs[0,0].scatter(bin_centers, som_outliers, c='0.3', s=20)
    axs[0,0].scatter(bin_centers, som_nn_out, c='0.4', alpha=0.7, s=20)
    axs[0,0].plot(bin_centers, umap_outliers, c='b', label='UMAP (LePHARE-trained)', ls='--')
    axs[0,0].scatter(bin_centers, umap_outliers, c='b', s=20)
    axs[0,0].set_ylabel(r'$f_\mathrm{outlier}$', fontsize=25)
    axs[0,0].tick_params(axis='y', labelsize=18)
    axs[0,0].tick_params(axis='x', labelsize=18)
    axs[0,0].set_ylim(0,fout_ymax)
    axs[0,0].set_xlim(bin_min,None)
    # MIDDLE LEFT
    axs[1,0].plot(bin_centers, som_nmads, c='0.4', alpha=0.7, ls='--')
    axs[1,0].scatter(bin_centers, som_nmads, c='0.4', alpha=0.7, s=20)
    axs[1,0].plot(bin_centers, umap_nmads, c='b', ls='--')
    axs[1,0].scatter(bin_centers, umap_nmads, c='b', s=20)
    axs[1,0].set_ylabel(r'$\sigma_\mathrm{NMAD}$', fontsize=25)
    axs[1,0].tick_params(axis='y', labelsize=18)
    axs[1,0].tick_params(axis='x', labelsize=18)
    axs[1,0].set_ylim(0,nmad_ymax)
    axs[1,0].set_xlim(bin_min,None)
    # BOTTOM LEFT
    axs[2,0].plot(bin_centers, som_biases, c='0.4', alpha=0.7, ls='--')
    axs[2,0].scatter(bin_centers, som_biases, c='0.4', alpha=0.7, s=20)
    axs[2,0].plot(bin_centers, umap_biases, c='b', ls='--')
    axs[2,0].scatter(bin_centers, umap_biases, c='b', s=20)
    axs[2,0].axhline(y = 0.0, color = 'k', linestyle = ':', lw=1.3, alpha=0.8)
    axs[2,0].set_ylim(-bias_ymax,bias_ymax)
    axs[2,0].set_xlim(bin_min,None)
    axs[2,0].set_ylabel('Bias ('+r'$\langle \frac{\Delta z}{1+z} \rangle$'+')', fontsize=25)
    axs[2,0].tick_params(axis='y', labelsize=18)
    axs[2,0].tick_params(axis='x', labelsize=18)
    axs[2,0].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)
    # UPPER RIGHT
    axs[0,1].plot(bin_centers, umap_outliers, c='b', ls='--')
    axs[0,1].scatter(bin_centers, umap_outliers, c='b', s=20)
    axs[0,1].plot(bin_centers_s, som_outliers_s, c='0.3', label='SOM (spec-'+r'$z$'+'-trained, all cells)')
    axs[0,1].plot(bin_centers_s, som_nn_out_s, c='0.4', alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, cells with training redshifts)')
    axs[0,1].plot(bin_centers_s, umap_outliers_s, c='b', label='UMAP (spec-'+r'$z$'+'-trained)')
    axs[0,1].fill_between(bin_centers_s, som_outliers_s, y2=som_nn_out_s, alpha=0.15, color='0.4')
    axs[0,1].scatter(bin_centers_s, som_outliers_s, c='0.3', s=20)
    axs[0,1].scatter(bin_centers_s, som_nn_out_s, c='0.4', alpha=0.7, s=20)
    axs[0,1].scatter(bin_centers_s, umap_outliers_s, c='b', s=20)
    axs[0,1].tick_params(axis='y', labelsize=18)
    axs[0,1].tick_params(axis='x', labelsize=18)
    axs[0,1].set_ylim(0,fout_ymax)
    axs[0,1].set_xlim(bin_min,None)
    # MIDDLE RIGHT
    axs[1,1].plot(bin_centers, umap_nmads, c='b', ls='--')
    axs[1,1].scatter(bin_centers, umap_nmads, c='b', s=20)
    axs[1,1].plot(bin_centers_s, som_nmads_s, c='0.4', alpha=0.7)
    axs[1,1].plot(bin_centers_s, umap_nmads_s, c='b')
    axs[1,1].scatter(bin_centers_s, som_nmads_s, c='0.4', alpha=0.7, s=20)
    axs[1,1].scatter(bin_centers_s, umap_nmads_s, c='b', s=20)
    axs[1,1].tick_params(axis='y', labelsize=18)
    axs[1,1].tick_params(axis='x', labelsize=18)
    axs[1,1].set_ylim(0,nmad_ymax)
    axs[1,1].set_xlim(bin_min,None)
    # BOTTOM RIGHT
    axs[2,1].plot(bin_centers, umap_biases, c='b', ls='--')
    axs[2,1].scatter(bin_centers, umap_biases, c='b', s=20)
    axs[2,1].plot(bin_centers_s, som_biases_s, c='0.4', alpha=0.7)
    axs[2,1].plot(bin_centers_s, umap_biases_s, c='b')
    axs[2,1].scatter(bin_centers_s, som_biases_s, c='0.4', alpha=0.7, s=18)
    axs[2,1].scatter(bin_centers_s, umap_biases_s, c='b', s=18)
    axs[2,1].axhline(y = 0.0, color = 'k', linestyle =':', lw=1.3, alpha=0.8)
    axs[2,1].set_ylim(-bias_ymax,bias_ymax)
    axs[2,1].set_xlim(bin_min,None)
    axs[2,1].tick_params(axis='y', labelsize=18)
    axs[2,1].tick_params(axis='x', labelsize=18)
    axs[2,1].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)
    fig.subplots_adjust(top=0.95)
    fig.legend(bbox_to_anchor=(0.03, 1.02, 1., .102), loc='center',
                        ncols=2, borderaxespad=0., fontsize=17)
    fig.tight_layout()
    #fig.suptitle('SOM vs UMAP: Statistics by Redshift', fontsize=14);
    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')

# create a plot similar to Figure 5, comparing the UMAP and input color space performance
def z_binned_umap_vs_colors(fullcat_lp, fullcat_spec, umap_test_lp, umap_test_spec, c_test_lp, c_test_spec, i_vals='TEST_Z', t_vals='lp_photoz', bin_min=0.0, bin_max=3.0, bin_size=0.2, filename=None):
    '''This function creates a multi-panel plot of the performance statistics binned by redshift, 
    for comparing the input seven-dimensional color space to the three-dimensional UMAP space.

    Parameters
    ----------
    fullcat_lp: pandas DataFrame
        DataFrame containing the full catalog of test objects, LePHARE-trained case
    fullcat_spec: pandas DataFrame
        DataFrame containing the full catalog of test objects, spec-z-trained case
    umap_test_lp: pandas DataFrame
        DataFrame containing the catalog of objects with UMAP-kNN-z redshift estimates, 
        LePHARE-trained case
    umap_test_spec: pandas DataFrame
        DataFrame containing the catalog of objects with UMAP-kNN-z redshift estimates, 
        spec-z-trained case
    c_test_lp: pandas DataFrame
        DataFrame containing the catalog of objects with colors-kNN-z redshift estimates, 
        LePHARE-trained case
    c_test_spec: pandas DataFrame
        DataFrame containing the catalog of objects with colors-kNN-z redshift estimates, 
        spec-z-trained case
    i_vals: string, default='TEST_Z'
        string corresponding to the label of the DataFrame column containing the 
        redshift estimates interpolated using either UMAP-kNN-z or SOM-z
    t_vals: string, default='lp_photoz'
        string corresponding to the label of the DataFrame column containing the 
        "true" redshifts
    bin_min: float, default=0.0
        the lower limit of the redshift binning, and the lower x-axis limit for plotting
    bin_max: float, default=3.0
        the upper limit of the redshift binning
    bin_size: float, default=0.2
        the width of the redshift bins
    filename: string, default=None
        optional filename, if !=None the plot will be saved according to the format in the 
        string, e.g. 'z_binned_umap_vs_colors.png'

    Returns
    ----------
    figure: the plot!
    optionally saved to directory if filename!=None
    '''
    # calculate z-binned stats for colors-kNN-z and UMAP-kNN-z
    bin_centers, lp_counts, lp_outliers, lp_nmads, lp_biases, lp_nn_out = binned_stats(fullcat=fullcat_lp, testcat=c_test_lp, 
                                                                            iv=i_vals, tv=t_vals, bin_var=t_vals, bin_min=bin_min, 
                                                                         bin_max=bin_max, bin_size=bin_size)
    bin_centers_s, counts_s, outliers_s, nmads_s, biases_s, nn_out_s = binned_stats(fullcat=fullcat_spec, testcat=c_test_spec, 
                                                                            iv=i_vals, tv=t_vals, bin_var=t_vals, bin_min=bin_min, 
                                                                         bin_max=bin_max, bin_size=bin_size)
    bin_centers, umap_counts, umap_outliers, umap_nmads, umap_biases, umap_nn_out = binned_stats(fullcat=fullcat_lp, testcat=umap_test_lp, 
                                                                         iv=i_vals, tv=t_vals, bin_var=t_vals, bin_min=bin_min, 
                                                                         bin_max=bin_max, bin_size=bin_size)
    bin_centers_s, umap_counts_s, umap_outliers_s, umap_nmads_s, umap_biases_s, umap_nn_out_s = binned_stats(fullcat=fullcat_spec, testcat=umap_test_spec, 
                                                                         iv=i_vals, tv=t_vals, bin_var=t_vals, bin_min=bin_min, 
                                                                         bin_max=bin_max, bin_size=bin_size)


    # create the figure
    fig, axs = plt.subplots(3, 1, figsize=(12,16))

    fout_ymax = 1.05*np.max([np.max(lp_outliers), np.max(outliers_s), np.max(umap_outliers), np.max(umap_outliers_s)])
    nmad_ymax = 1.05*np.max([np.max(lp_nmads), np.max(nmads_s), np.max(umap_nmads), np.max(umap_nmads_s)])
    bias_ymax = 1.05*np.max([np.max(lp_biases), np.max(biases_s), np.max(umap_biases), np.max(umap_biases_s)])

    # UPPER LEFT
    axs[0].plot(bin_centers, outliers_s, c='0.7', label='Color Space (CSRC-trained)')
    axs[0].plot(bin_centers, lp_outliers, c='0.3', label='Color Space (LePhare-trained)')
    axs[0].plot(bin_centers, umap_outliers_s, c='cyan', label='UMAP (CSRC-trained)')
    axs[0].plot(bin_centers, umap_outliers, c='b', label='UMAP (LePhare-trained)')
    axs[0].scatter(bin_centers, outliers_s, c='0.7', s=20)
    axs[0].scatter(bin_centers, lp_outliers, c='0.3', s=20)
    axs[0].scatter(bin_centers, umap_outliers_s, c='cyan', s=20)
    axs[0].scatter(bin_centers, umap_outliers, c='b', s=20)
    axs[0].set_ylabel(r'$f_\mathrm{outlier}$', fontsize=25)
    axs[0].tick_params(axis='y', labelsize=18)
    axs[0].tick_params(axis='x', labelsize=18)
    axs[0].set_ylim(0,fout_ymax)
    axs[0].set_xlim(0,None)
    axs[0].legend(fontsize=18)

    # MIDDLE LEFT
    axs[1].plot(bin_centers, nmads_s, c='0.7', label='Color Space (CSRC-trained)')
    axs[1].plot(bin_centers, lp_nmads, c='0.3', label='Color Space (LePhare-trained)')
    axs[1].plot(bin_centers, umap_nmads_s, c='cyan', label='UMAP (CSRC-trained)')
    axs[1].plot(bin_centers, umap_nmads, c='b', label='UMAP (LePhare-trained)')
    axs[1].scatter(bin_centers, nmads_s, c='0.7', s=20)
    axs[1].scatter(bin_centers, lp_nmads, c='0.3', s=20)
    axs[1].scatter(bin_centers, umap_nmads_s, c='cyan', s=20)
    axs[1].scatter(bin_centers, umap_nmads, c='b', s=20)
    axs[1].set_ylabel(r'$\sigma_\mathrm{NMAD}$', fontsize=25)
    axs[1].tick_params(axis='y', labelsize=18)
    axs[1].tick_params(axis='x', labelsize=18)
    axs[1].set_ylim(0,nmad_ymax)
    axs[1].set_xlim(0,None)

    # BOTTOM LEFT
    axs[2].plot(bin_centers, biases_s, c='0.7', label='Color Space (CSRC-trained)')
    axs[2].plot(bin_centers, lp_biases, c='0.3', label='Color Space (LePhare-trained)')
    axs[2].plot(bin_centers, umap_biases_s, c='cyan', label='UMAP (CSRC-trained)')
    axs[2].plot(bin_centers, umap_biases, c='b', label='UMAP (LePhare-trained)')
    axs[2].scatter(bin_centers, biases_s, c='0.7', s=20)
    axs[2].scatter(bin_centers, lp_biases, c='0.3', s=20)
    axs[2].scatter(bin_centers, umap_biases_s, c='cyan', s=20)
    axs[2].scatter(bin_centers, umap_biases, c='b', s=20)
    axs[2].axhline(y = 0.0, color = 'k', linestyle = '-.', lw=1.3, alpha=0.6)
    axs[2].set_ylim(-bias_ymax,bias_ymax)
    axs[2].set_xlim(0,None)
    axs[2].set_ylabel('Bias ('+r'$\langle \frac{\Delta z}{1+z} \rangle$'+')', fontsize=25)
    axs[2].tick_params(axis='y', labelsize=18)
    axs[2].tick_params(axis='x', labelsize=18)
    axs[2].set_xlabel(r'$z_\mathrm{phot}$', fontsize=25)

    fig.tight_layout()
    #fig.suptitle('SOM vs UMAP: Statistics by Redshift', fontsize=14);
    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')
