# umapz: functions for using UMAP (or SOM) dimensionality reduced 
# color space to interpolate redshifts, and to test the 
# quality of these mappings and interpolations
import sys
#print('x')
#print(sys.executable)
#print('x')
import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm
from matplotlib.transforms import Bbox
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
#import seaborn as sns
#sns.set(style='white', rc={'figure.figsize':(14, 12), 'animation.html': 'html5'})
import colorcet as cc
import subprocess
import copy
import glob

from sklearn.model_selection import KFold
import pickle
from minisom import MiniSom
from pathlib import Path


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

def data_split(photcat, speccat, valid_size, rs=41):
    '''
    Extract a validation set from the sample for embedding/interpolator optimization, then from the remaining
    objects create train and test sets for the LePHARE-trained and spec-z-trained cases. 

    Parameters
    ----------
    photcat: pandas DataFrame
        the entire photometric dataset, must contain all columns necessary to perform the split,
        and must contain the spec-z objects (but not their spec-z values)
    speccat: pandas DataFrame
        the dataset with the spec-z's
    valid_size: integer
        size of the validation set
    
    Returns
    ----------
    data: dictionary
        dictionary containing the dataset split results.
        keys 'validation', 'train_lp', 'test_lp', 'train_spec', 'test_spec'
    '''
    no_spec = photcat[~photcat['ID'].isin(speccat['ID'])]
    valid = no_spec.sample(n=valid_size, random_state=rs)
    traintest = photcat[~photcat['ID'].isin(valid['ID'])]
    train_spec = pd.merge(speccat, traintest[['ID']],#, 'UMAP3D-1', 'UMAP3D-2', 'UMAP3D-3', 'SOM-1', 'SOM-2']], 
                      left_on='ID', right_on='ID', how='left')
    test_spec = traintest[~traintest['ID'].isin(speccat['ID'])]
    train_lp = traintest.copy().sample(n=len(train_spec), random_state=rs)
    test_lp = traintest.copy()[~traintest['ID'].isin(train_lp['ID'])]

    return {'valid':valid, 'train_lp':train_lp, 'test_lp':test_lp, 'train_spec':train_spec, 'test_spec':test_spec}

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

def calc_stdev(i_vals, t_vals):
    '''This function calculates the standard deviation between 
    the UMAP-interpolated/SOM-assigned and truth redshift values for a given test sample

    Parameters
    ----------
    i_vals: pandas DataFrame column of floats
        the UMAP-interpolated / SOM-assigned redshift values
    t_vals: pandas DataFrame column of floats
        the "truth" redshift values (LePhare or spectroscopic) of the same sources

    Returns
    ---------- 
    float: standard deviation
    '''
    delta_z = (i_vals - t_vals) / (1 + t_vals)
    return np.std(delta_z)

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
    z_true: string
        redshift label, e.g., 'lp_photoz' or 'specz'

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

def som_interpolator(df1, df2, map='SOM', dim=2, nn=50, calc='median', z_true='lp_photoz'):    
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
    
    if calc == 'mean':
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

def som_rematch(df1, df2, map, zlabel, nn=50, calc='median'):
    '''
    this function replaces X
    '''
    coords = [map+'-1', map+'-2']
    # get mean colors in SOM cells
    colors=['u-g', 'g-r', 'r-i', 'i-z', 'z-y', 'y-J', 'J-H']
    mean_colors = df1.groupby(coords)[colors].mean().reset_index()
    if calc=='median':
        mean_colors[zlabel] = df1.groupby(coords)[zlabel].median().reset_index()[zlabel]
    elif calc=='mean':
        mean_colors[zlabel] = df1.groupby(coords)[zlabel].mean().reset_index()[zlabel]
    # match sources in other catalog to nearest neighbor in mean_colors

    # assign redshifts to all objects based on the nearest 
    known_z = mean_colors[zlabel]
    known_map = mean_colors[colors]
    nn_model = neighbors.NearestNeighbors(n_neighbors=1, p=2)
    nn_model.fit(known_map)
    # identify 
    df2_ = df2.copy()[~df2['ID'].isin(df1['ID'])]
    unknown_map = df2_[colors]
    distances, indices = nn_model.kneighbors(unknown_map)
    interpolated_z = known_z.values[indices]
    df2_['TEST_Z'] = interpolated_z

    # separate galaxies in the test set that are not in the training set
    #df3_ = df2.copy()[~df2['ID'].isin(df1['ID'])]
    # standard SOM-z interpolation
    df3 = interpolator(df1=df1, df2=df2_, map=map, dim=2, nn=nn, calc='median', z_true=zlabel)
    #df3 = umapz2.interpolator(df1=paper_spec95, df2=df3_, map=map, dim=2, nn=nn, calc='median', z_true=zlabel)
    # identify objects in the color-matched test set that are missing from the SOM-z-interpolated test set
    df4 = df2_.copy()[~df2_['ID'].isin(df3['ID'])]
    # combine the SOM-z set with the objects missing from it (including their color-matched redshifts)
    # + track which objects have redshifts from re-matched cells
    df3['cellmatch'] = np.full(shape=len(df3), fill_value=1)
    df4['cellmatch'] = np.full(shape=len(df4), fill_value=2)
    full_som_test = pd.concat([df3, df4], axis=0)

    return full_som_test


def binned_stats_old(fullcat, testcat, iv, tv, bin_var, bin_min, bin_max, bin_size):
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
    stdevs = []
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
        stdevs.append(calc_stdev(i_vals=testcat_i[iv], t_vals=testcat_i[tv]))
        biases.append(calc_bias(i_vals=testcat_i[iv], t_vals=testcat_i[tv]))

    return {'bin_centers':bin_centers, 'counts':counts, 'outliers':outliers, 'nmads':nmads, 
            'stdevs':stdevs, 'biases':biases, 'nn_outliers':nn_outliers}

def binned_stats(fullcat, testcat, iv, tv, bin_var, bin_min, bin_max, bin_size):
    '''Function (used as helper function in z_binned_stats and z_binned_umap_vs_colors) that 
    calculates the bin centers as well as the number of objects (counts), fraction of outliers, 
    NMAD, bias, and fraction of outliers (trained cells only) for each bin

    Parameters
    ----------
    fullcat: pandas DataFrame
        DataFrame containing the full test/target set catalog --> ONLY USED TO CALCULATE "MISSING"
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
    outliers_p = []
    m_outliers = []
    rm_outliers = []
    m_nmads = []
    rm_nmads = []
    m_biases = []
    rm_biases = []
    m_stdevs = []
    rm_stdevs = []

    rm_only_outliers = []
    rm_only_nmads = []
    rm_only_biases = []

    ## need additional final dictionary entries for outliers, nmad, and biases with som_rematch
    ## can keep storing all redshift estimates as 'TEST_Z'; track which objects have rematched-cell-z's
    ## --> use count as "missing", exclude from "nn", use all for new curve (all test objects including re-matching)
    ## --> when storing redshifts in som_rematch, add column 'cell_match' with value 1 or 2 (2 for re-matched test objects)

    bin_centers = np.arange(bin_min+bin_size/2, bin_max+bin_size/2, bin_size)
    for i, bc in enumerate(bin_centers):
        low_end = bc - bin_size/2
        high_end = bc + bin_size/2
        testcat_rm = testcat[(testcat[bin_var]>=low_end) & (testcat[bin_var]<high_end)]
        testcat_m = testcat[(testcat[bin_var]>=low_end) & (testcat[bin_var]<high_end) & (testcat['cellmatch']==1)]
        testcat_rm_only = testcat[(testcat[bin_var]>=low_end) & (testcat[bin_var]<high_end) & (testcat['cellmatch']==2)]
  

        counts.append(len(testcat_m))
        outliers_p.append(f_outlier(i_vals=testcat_m[iv], t_vals=testcat_m[tv], missing=len(testcat_rm)-len(testcat_m), threshold=0.15))
        rm_outliers.append(f_outlier(i_vals=testcat_rm[iv], t_vals=testcat_rm[tv], missing=0., threshold=0.15))
        m_outliers.append(f_outlier(i_vals=testcat_m[iv], t_vals=testcat_m[tv], missing=0., threshold=0.15))
        m_nmads.append(calc_nmad(i_vals=testcat_m[iv], t_vals=testcat_m[tv]))
        rm_nmads.append(calc_nmad(i_vals=testcat_rm[iv], t_vals=testcat_rm[tv]))
        m_biases.append(calc_bias(i_vals=testcat_m[iv], t_vals=testcat_m[tv]))
        rm_biases.append(calc_bias(i_vals=testcat_rm[iv], t_vals=testcat_rm[tv]))
        m_stdevs.append(calc_stdev(i_vals=testcat_m[iv], t_vals=testcat_m[tv]))
        rm_stdevs.append(calc_stdev(i_vals=testcat_rm[iv], t_vals=testcat_rm[tv]))

        rm_only_outliers.append(f_outlier(i_vals=testcat_rm_only[iv], t_vals=testcat_rm_only[tv], missing=0., threshold=0.15))
        rm_only_nmads.append(calc_nmad(i_vals=testcat_rm_only[iv], t_vals=testcat_rm_only[tv]))
        rm_only_biases.append(calc_bias(i_vals=testcat_rm_only[iv], t_vals=testcat_rm_only[tv]))

    return {'bin_centers':bin_centers, 'counts':counts, 'outliers_p':outliers_p, 
            'm_outliers':m_outliers, 'rm_outliers':rm_outliers, 'm_nmads':m_nmads, 
            'rm_nmads':rm_nmads, 'm_biases':m_biases, 'rm_biases':rm_biases,
            'm_stdevs':m_stdevs, 'rm_stdevs':rm_stdevs,
            'rm_only_outliers':rm_only_outliers, 'rm_only_nmads':rm_only_nmads, 'rm_only_biases':rm_only_biases}

def binned_stats_ensemble2(fullcats, testcats, iv, tv, bin_var, bin_min, bin_max, bin_size):
    """
    Run binned_stats on multiple (fullcat, testcat) realizations
    and return mean and std of each metric per bin.

    Parameters
    ----------
    fullcats: 
        x
    testcats: 
        x
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
    """

    assert len(fullcats) == len(testcats), "fullcats and testcats must have same length"

    # storage: one list per realization
    counts_all = []
    outliers_m_all = []
    nmads_m_all = []
    stdevs_m_all = []
    biases_m_all = []
    outliers_rm_all = []
    nmads_rm_all = []
    stdevs_rm_all = []
    biases_rm_all = []

    for fullcat, testcat in zip(fullcats, testcats):
        bs = binned_stats(
            fullcat, testcat,
            iv, tv, bin_var,
            bin_min, bin_max, bin_size)

        counts_all.append(bs['counts'])
        outliers_m_all.append(bs['m_outliers'])
        nmads_m_all.append(bs['m_nmads'])
        stdevs_m_all.append(bs['mstdevs'])
        biases_m_all.append(bs['m_biases'])
        outliers_rm_all.append(bs['rm_outliers'])
        nmads_rm_all.append(bs['rm_nmads'])
        stdevs_rm_all.append(bs['stdevs'])
        biases_rm_all.append(bs['rm_biases'])

    # convert to arrays: shape = (n_realizations, n_bins)
    counts_all = np.asarray(counts_all)
    outliers_m_all = np.asarray(outliers_m_all)
    nmads_m_all = np.asarray(nmads_m_all)
    stdevs_m_all = np.asarray(stdevs_m_all)
    biases_m_all = np.asarray(biases_m_all)
    outliers_rm_all = np.asarray(outliers_rm_all)
    nmads_rm_all = np.asarray(nmads_rm_all)
    stdevs_rm_all = np.asarray(stdevs_rm_all)
    biases_rm_all = np.asarray(biases_rm_all)

    results = {
        "bin_centers": bs['bin_centers'],

        "counts_mean": counts_all.mean(axis=0),
        "counts_std": counts_all.std(axis=0),

        "outliers_m_mean": outliers_m_all.mean(axis=0),
        "outliers_m_std": outliers_m_all.std(axis=0),
        "outliers_m_range": outliers_m_all.max(axis=0) - outliers_m_all.min(axis=0),

        "nmads_m_mean": nmads_m_all.mean(axis=0),
        "nmads_m_std": nmads_m_all.std(axis=0),
        "nmads_m_range": nmads_m_all.max(axis=0) - nmads_m_all.min(axis=0),

        "stdevs_m_mean": stdevs_m_all.mean(axis=0),
        "stdevs_m_std": stdevs_m_all.std(axis=0),
        "stdevs_m_range": stdevs_m_all.max(axis=0) - stdevs_m_all.min(axis=0),

        "biases_m_mean": biases_m_all.mean(axis=0),
        "biases_m_std": biases_m_all.std(axis=0),
        "biases_m_range": biases_m_all.max(axis=0) - biases_m_all.min(axis=0),

        "outliers_rm_mean": outliers_rm_all.mean(axis=0),
        "outliers_rm_std": outliers_rm_all.std(axis=0),
        "outliers_rm_range": outliers_rm_all.max(axis=0) - outliers_rm_all.min(axis=0),

        "nmads_rm_mean": nmads_rm_all.mean(axis=0),
        "nmads_rm_std": nmads_rm_all.std(axis=0),
        "nmads_rm_range": nmads_rm_all.max(axis=0) - nmads_rm_all.min(axis=0),

        "stdevs_rm_mean": stdevs_rm_all.mean(axis=0),
        "stdevs_rm_std": stdevs_rm_all.std(axis=0),
        "stdevs_rm_range": stdevs_rm_all.max(axis=0) - stdevs_rm_all.min(axis=0),

        "biases_rm_mean": biases_rm_all.mean(axis=0),
        "biases_rm_std": biases_rm_all.std(axis=0),
        "biases_rm_range": biases_rm_all.max(axis=0) - biases_rm_all.min(axis=0)
    }

    return results    


def binned_stats_ensemble(fullcats, testcats, iv, tv, bin_var, bin_min, bin_max, bin_size):
    """
    Run binned_stats on multiple (fullcat, testcat) realizations
    and return mean and std of each metric per bin.

    Parameters
    ----------
    fullcats: 
        x
    testcats: 
        x
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
    """

    assert len(fullcats) == len(testcats), "fullcats and testcats must have same length"

    # storage: one list per realization
    counts_all = []
    outliers_all = []
    nn_outliers_all = []
    nmads_all = []
    stdevs_all = []
    biases_all = []

    for fullcat, testcat in zip(fullcats, testcats):
        bso = binned_stats_old(
            fullcat, testcat,
            iv, tv, bin_var,
            bin_min, bin_max, bin_size)

        counts_all.append(bso['counts'])
        outliers_all.append(bso['outliers'])
        nmads_all.append(bso['nmads'])
        stdevs_all.append(bso['stdevs'])
        biases_all.append(bso['biases'])
        nn_outliers_all.append(bso['nn_outliers'])

    # convert to arrays: shape = (n_realizations, n_bins)
    counts_all = np.asarray(counts_all)
    outliers_all = np.asarray(outliers_all)
    nmads_all = np.asarray(nmads_all)
    stdevs_all = np.asarray(stdevs_all)
    biases_all = np.asarray(biases_all)
    nn_outliers_all = np.asarray(nn_outliers_all)

    results = {
        "bin_centers": bso['bin_centers'],

        "counts_mean": counts_all.mean(axis=0),
        "counts_std": counts_all.std(axis=0),

        "outliers_mean": outliers_all.mean(axis=0),
        "outliers_std": outliers_all.std(axis=0),
        "outliers_range": outliers_all.max(axis=0) - outliers_all.min(axis=0),

        "nn_outliers_mean": nn_outliers_all.mean(axis=0),
        "nn_outliers_std": nn_outliers_all.std(axis=0),
        "nn_outliers_range": nn_outliers_all.max(axis=0) - nn_outliers_all.min(axis=0),

        "nmads_mean": nmads_all.mean(axis=0),
        "nmads_std": nmads_all.std(axis=0),
        "nmads_range": nmads_all.max(axis=0) - nmads_all.min(axis=0),

        "stdevs_mean": stdevs_all.mean(axis=0),
        "stdevs_std": stdevs_all.std(axis=0),
        "stdevs_range": stdevs_all.max(axis=0) - stdevs_all.min(axis=0),

        "biases_mean": biases_all.mean(axis=0),
        "biases_std": biases_all.std(axis=0),
        "biases_range": biases_all.max(axis=0) - biases_all.min(axis=0),
    }

    return results


def wrapped_results(n_states, data, nn, map, tv):
    '''
    Wrapper for binned_stats ensemble that synthesizes the generation of the n_states randomized samples, 
    the interpolation using a specified nn argument, and the calculation of the results. 

    Parameters:
    ----------
    n_states: int
        the number of random splits of the data to test
    data: DataFrame
        the DataFrame containing the spectroscopic data w/ UMAP coords
    nn: int
        the k value for UMAP-kNN-z
    '''

    dummy = {'train':[0], 'test':[0]}
    sets = {i: copy.deepcopy(dummy) for i in range(1, n_states+1)}

    # splits
    # randomize the order of the dataset
    states = list(range(1,n_states+1))
    ts = 2_900
    for state in states:
        df_random = data.sample(n=len(data), random_state=state)
        train = df_random.copy()[:len(data)-ts]
        test_ = df_random.copy()[len(data)-ts:len(data)]
        test = interpolator(df1=train, df2=test_, map=map, dim=3, nn=nn, calc='median', z_true=tv)
        sets[state]['train'] = train.copy()
        sets[state]['test'] = test.copy()

    testcats_ = []
    for i in range(1,n_states+1): testcats_.append(sets[i]['test'].copy())
    #fullcats8020 = [spec.copy()] * len(testcats8020)

    results = binned_stats_ensemble(
        testcats_,
        testcats_,
        iv="TEST_Z",
        tv="lp_photoz",
        bin_var="lp_photoz",
        bin_min=0.0,
        bin_max=3.0,
        bin_size=0.2,
    )

    return results


def wrapped_global(n_states, data, nn, map='UMAP41', tv='specz'):
    '''
    Function that synthesizes the generation of the n_states randomized samples, 
    the interpolation using a specified nn argument, and the calculation of the results. 
    Yields the global metrics and standard deviations over n_states runs.

    Parameters:
    ----------
    n_states: int
        the number of random splits of the data to test
    data: DataFrame
        the DataFrame containing the spectroscopic data w/ UMAP coords
    nn: int
        the k value for UMAP-kNN-z

    Returns:
    ----------

    '''
    dummy = {'train':[0], 'test':[0]}
    sets = {i: copy.deepcopy(dummy) for i in range(1, n_states+1)}

    # splits
    # randomize the order of the dataset
    states = list(range(1,n_states+1))
    ts = 2_900
    counts_all = []
    outliers_all = []
    nn_outliers_all = []
    nmads_all = []
    biases_all = []

    for state in states:
        df_random = data.sample(n=len(data), random_state=state)
        train = df_random.copy()[:len(data)-ts]
        test_ = df_random.copy()[len(data)-ts:len(data)]
        test = interpolator(df1=train, df2=test_, map=map, dim=3, nn=nn, calc='median', z_true=tv)
        # map^ was previously hardcoded as 'UMAP41', if this breaks that may be the issue
        sets[state]['train'] = train.copy()
        sets[state]['test'] = test.copy()
        testcat = sets[state]['test'].copy()
        counts_all.append(len(testcat))
        outliers_all.append(f_outlier(i_vals=testcat['TEST_Z'], t_vals=testcat[tv], missing=0, threshold=0.15))
        nn_outliers_all.append(f_outlier(i_vals=testcat['TEST_Z'], t_vals=testcat[tv], missing=0., threshold=0.15))
        nmads_all.append(calc_nmad(i_vals=testcat['TEST_Z'], t_vals=testcat[tv]))
        biases_all.append(calc_bias(i_vals=testcat['TEST_Z'], t_vals=testcat[tv]))

    # convert to arrays: shape = (n_realizations, n_bins)
    counts_all = np.asarray(counts_all)
    outliers_all = np.asarray(outliers_all)
    nmads_all = np.asarray(nmads_all)
    biases_all = np.asarray(biases_all)
    nn_outliers_all = np.asarray(nn_outliers_all)

    results = {
    "counts": counts_all.mean(axis=0),
    "nn": nn,

    "outliers_mean": outliers_all.mean(axis=0),
    "outliers_std": outliers_all.std(axis=0),

    "nn_outliers_mean": nn_outliers_all.mean(axis=0),
    "nn_outliers_std": nn_outliers_all.std(axis=0),

    "nmads_mean": nmads_all.mean(axis=0),
    "nmads_std": nmads_all.std(axis=0),

    "biases_mean": biases_all.mean(axis=0),
    "biases_std": biases_all.std(axis=0),
    }

    return results

def crossvalidate_global(folds, data, nn, map='UMAP41', tv='specz', dim=3, calc='median', rematch=False, colors=False, p=2):
    '''
    Function that synthesizes the generation of the n_states randomized samples, 
    the interpolation using a specified nn argument, and the calculation of the results. 
    Yields the mean global metrics and their ranges /2*sqrt(folds) over n_states runs.

    Parameters
    ----------
    n_states: int
        the number of random splits of the data to test
    data: DataFrame
        the DataFrame containing the spectroscopic/validation data w/ UMAP coords
    nn: int
        the k value for UMAP-kNN-z
    map:
    tv: string
        the column name containing the "true" z-values
    Returns
    ----------

    '''
    dummy = {'train':[0], 'test':[0]}
    sets = {i: copy.deepcopy(dummy) for i in range(1, folds+1)}

    # 5-fold cross validation
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    counts_all = []
    outliers_all = []
    nmads_all = []
    biases_all = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        train = data.iloc[train_idx].copy()
        test_ = data.iloc[test_idx].copy()

        if rematch==True:
            test = som_rematch(df1=train, df2=test_, map=map, zlabel=tv, nn=nn, calc=calc)
        elif colors==True:
            test = interpolator_colors(df1=train, df2=test_, coords=['u-g', 'g-r', 'r-i', 'i-z', 
                                        'z-y', 'y-J', 'J-H'], nn=nn, calc=calc, z_true=tv, p=p)
        else:
            test = interpolator(df1=train, df2=test_, map=map, dim=dim, nn=nn, calc=calc, z_true=tv)

        sets[fold]['train'] = train.copy()
        sets[fold]['test'] = test.copy()

        testcat = test.copy()

        counts_all.append(len(testcat))

        outliers_all.append(f_outlier(i_vals=testcat['TEST_Z'], t_vals=testcat[tv], missing=0, threshold=0.15))

        nmads_all.append(calc_nmad(i_vals=testcat['TEST_Z'],t_vals=testcat[tv]))

        biases_all.append(calc_bias(i_vals=testcat['TEST_Z'], t_vals=testcat[tv]))

    # Convert to arrays
    counts_all = np.asarray(counts_all)
    outliers_all = np.asarray(outliers_all)
    nmads_all = np.asarray(nmads_all)
    biases_all = np.asarray(biases_all)

    results = {
        "counts_mean": counts_all.mean(axis=0),
        "nn": nn,

        "outliers_mean": outliers_all.mean(axis=0),
        "outliers_range": (outliers_all.max(axis=0) - outliers_all.min(axis=0))/(2*np.sqrt(folds)),

        "nmads_mean": nmads_all.mean(axis=0),
        "nmads_range": (nmads_all.max(axis=0) - nmads_all.min(axis=0))/(2*np.sqrt(folds)),

        "biases_mean": biases_all.mean(axis=0),
        "biases_range": (biases_all.max(axis=0) - biases_all.min(axis=0))/(2*np.sqrt(folds)),
    }

    return results, sets

def crossvalidate_binned(folds, data, nn, map='UMAP41', tv='specz', dim=3, rematch=False, colors=False, p=2):
    '''
    Analogous to wrapped_results (wrapper for binned_stats ensemble), but uses the cross-validation splits
    rather than random splits.
    '''
    dummy = {'train':[0], 'test':[0]}
    sets = {i: copy.deepcopy(dummy) for i in range(1, folds+1)}

    # 5-fold cross validation
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        train = data.iloc[train_idx].copy()
        test_ = data.iloc[test_idx].copy()

        if rematch==True:
            test = som_rematch(df1=train, df2=test_, map=map, zlabel=tv, nn=nn)
        elif colors==True:
            test = interpolator_colors(df1=train, df2=test_, coords=['u-g', 'g-r', 'r-i', 'i-z', 
                                        'z-y', 'y-J', 'J-H'], nn=nn, calc='median', z_true=tv, p=p)
        else:
            test = interpolator(df1=train, df2=test_, map=map, dim=dim, nn=nn, calc='median', z_true=tv)

        sets[fold]['train'] = train.copy()
        sets[fold]['test'] = test.copy()

    testcats_ = []
    for i in range(1,folds+1): testcats_.append(sets[i]['test'].copy())

    results = binned_stats_ensemble(
        testcats_,
        testcats_,
        iv="TEST_Z",
        tv="lp_photoz",
        bin_var="lp_photoz",
        bin_min=0.0,
        bin_max=3.0,
        bin_size=0.2,
    )

    return results, sets

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

def train_soms(data, sizes, sigmas, learning_rates, n_iter=500_000, rs=42, n_colors=7, path_pck='/Users/finianashmead/Desktop/NEWMANGROUP/path_pck2', metric='euclidean'):
    '''
    Train numerous SOM embeddings for a specified dataset, with one or several sizes (width,height), 
    sigmas, learning rates.

    Parameters
    ----------
    data: pandas DataFrame
        the data!
    sizes: list of tuples of integers
        the widths and heights of the SOM grids
    sigmas: list of floats
        the SOM hyperparameter sigma
    learning_rates: list of floats
        the SOM hyperparameter learning_rate
    n_iter: integer
        the numbers of iterations
    rs: integer
        the SOM random state
    n_colors: integer
        the number of colors in the color space
    path_pck: string
        the path to the directory in which all of the embeddings generated 

    Returns
    ----------
    Data: pandas DataFrame 
        contains the resulting SOM coordinates in the format:
        'SOM_'+str(n_som_xy[0])+'_'+str(n_som_xy[1])+'_s'+str(sigma)+'_lr'+str(lr)+'-1'
        'SOM_'+str(n_som_xy[0])+'_'+str(n_som_xy[1])+'_s'+str(sigma)+'_lr'+str(lr)+'-2'

    Print messages show progress while running, the results are also written out into the pack_pck 
    directory in the format:
    'som_'+str(n_som_xy[0])+'_'+str(n_som_xy[1])+'_s'+str(sigma)+'_lr'+str(lr)+'_n'+str(n_iter)
    '''
    data2 = data.copy()
    data1 = data.iloc[:, 66:73].values

    for size in sizes:
        n_som_xy = size
        for sigma in sigmas:
            for lr in learning_rates:

                som_model_id = 'som_'+str(n_som_xy[0])+'_'+str(n_som_xy[1])+'_s'+str(sigma)+'_lr'+str(lr)+'_n'+str(n_iter)
                path_pck = Path(path_pck)
                path_pck_som = path_pck / som_model_id
                filename_som_model = path_pck_som / f'{som_model_id}.pck'
                filename_som_obj_cell_coord = path_pck_som / f'som_obj_cell_coord_{som_model_id}.pck'
                filename_som_cell_counts = path_pck_som / f'som_cell_counts_{som_model_id}.pck'

                if filename_som_model.exists():
                    with open(filename_som_model, 'rb') as infile:
                        som = pickle.load(infile)

                    with open(filename_som_obj_cell_coord, 'rb') as infile:
                        obj_cell_coord = pickle.load(infile)

                    with open(filename_som_cell_counts, 'rb') as infile:
                        cell_counts = pickle.load(infile)

                else:
                    # create directories here
                    os.mkdir(path_pck_som)
                    som = MiniSom(n_som_xy[0], n_som_xy[1], n_colors, sigma=sigma, learning_rate=lr, random_seed=rs, activation_distance=metric)
                    som.train(data1, num_iteration=n_iter)
                    obj_cell_coord, cell_counts = utils.objects_per_cell(som, data1, n_som_xy)

                    with open(filename_som_model, 'wb') as outfile:
                        pickle.dump(som, outfile)

                    with open(filename_som_obj_cell_coord, 'wb') as outfile:
                        pickle.dump(obj_cell_coord, outfile)

                    with open(filename_som_cell_counts, 'wb') as outfile:
                        pickle.dump(cell_counts, outfile)

                # convert the SOM coordinates of each object into a numpy array
                occ_array = np.array(obj_cell_coord)
                # similarly to with the UMAP embedding, save the SOM coordinates of each object to the DataFrame
                data2['SOM_'+str(n_som_xy[0])+'_'+str(n_som_xy[1])+'_s'+str(sigma)+'_lr'+str(lr)+'-1'] = occ_array[:,0].copy()
                data2['SOM_'+str(n_som_xy[0])+'_'+str(n_som_xy[1])+'_s'+str(sigma)+'_lr'+str(lr)+'-2'] = occ_array[:,1].copy()

                print('Done '+str(n_som_xy[0])+'_'+str(n_som_xy[1])+'_s'+str(sigma)+'_lr'+str(lr)+'_n'+str(n_iter))       
    print()
    print('DONE EMBEDDING')
    return data2

def train_som(data, size, sigma, learning_rate, n_iter=500_000, rs=42, n_colors=7, path_pck='/Users/finianashmead/Desktop/NEWMANGROUP/path_pck3', metric='euclidean'):
    '''
    Train numerous SOM embeddings for a specified dataset, with one or several sizes (width,height), 
    sigmas, learning rates.

    Parameters
    ----------
    data: pandas DataFrame
        the data!
    sizes: list of tuples of integers
        the widths and heights of the SOM grids
    sigmas: list of floats
        the SOM hyperparameter sigma
    learning_rates: list of floats
        the SOM hyperparameter learning_rate
    n_iter: integer
        the numbers of iterations
    rs: integer
        the SOM random state
    n_colors: integer
        the number of colors in the color space
    path_pck: string
        the path to the directory in which all of the embeddings generated 

    Returns
    ----------
    Data: pandas DataFrame 
        contains the resulting SOM coordinates in the format:
        'SOM_'+str(n_som_xy[0])+'_'+str(n_som_xy[1])+'_s'+str(sigma)+'_lr'+str(lr)+'-1'
        'SOM_'+str(n_som_xy[0])+'_'+str(n_som_xy[1])+'_s'+str(sigma)+'_lr'+str(lr)+'-2'

    Print messages show progress while running, the results are also written out into the pack_pck 
    directory in the format:
    'som_'+str(n_som_xy[0])+'_'+str(n_som_xy[1])+'_s'+str(sigma)+'_lr'+str(lr)+'_n'+str(n_iter)
    '''
    data2 = data.copy()
    data1 = data.iloc[:, 66:73].values

    som_model_id = 'som_'+str(size[0])+'_'+str(size[1])+'_s'+str(sigma)+'_lr'+str(learning_rate)+'_n'+str(n_iter)+'_rs'+str(rs)
    path_pck = Path(path_pck)
    path_pck_som = path_pck / som_model_id
    filename_som_model = path_pck_som / f'{som_model_id}.pck'
    filename_som_obj_cell_coord = path_pck_som / f'som_obj_cell_coord_{som_model_id}.pck'
    filename_som_cell_counts = path_pck_som / f'som_cell_counts_{som_model_id}.pck'

    if filename_som_model.exists():
        with open(filename_som_model, 'rb') as infile:
            som = pickle.load(infile)

        with open(filename_som_obj_cell_coord, 'rb') as infile:
            obj_cell_coord = pickle.load(infile)

        with open(filename_som_cell_counts, 'rb') as infile:
            cell_counts = pickle.load(infile)

    else:
        # create directories here
        os.mkdir(path_pck_som)
        som = MiniSom(size[0], size[1], n_colors, sigma=sigma, learning_rate=learning_rate, random_seed=rs, activation_distance=metric)
        som.train(data1, num_iteration=n_iter)
        obj_cell_coord, cell_counts = utils.objects_per_cell(som, data1, size)

        with open(filename_som_model, 'wb') as outfile:
            pickle.dump(som, outfile)

        with open(filename_som_obj_cell_coord, 'wb') as outfile:
            pickle.dump(obj_cell_coord, outfile)

        with open(filename_som_cell_counts, 'wb') as outfile:
            pickle.dump(cell_counts, outfile)

    # convert the SOM coordinates of each object into a numpy array
    occ_array = np.array(obj_cell_coord)
    # similarly to with the UMAP embedding, save the SOM coordinates of each object to the DataFrame
    data2['SOM_'+str(size[0])+'_'+str(size[1])+'_s'+str(sigma)+'_lr'+str(learning_rate)+'-1'] = occ_array[:,0].copy()
    data2['SOM_'+str(size[0])+'_'+str(size[1])+'_s'+str(sigma)+'_lr'+str(learning_rate)+'-2'] = occ_array[:,1].copy()

    print('Done '+str(size[0])+'_'+str(size[1])+'_s'+str(sigma)+'_lr'+str(learning_rate)+'_n'+str(n_iter))       
    print()
    print('DONE EMBEDDING')
    return data2

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

def plot_phot_som(phot_df, map, size, filename=None):
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
    3-column SOM plot
    counts_all, counts_lp, counts_s
    '''
    df1 = phot_df.copy()
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

    data_ = (np.log10(cell_counts_50), z_median_cell_50, ssfr_median_cell_50)


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
    n_cols = 3
    width, height = 0.25, 0.25
    x_spacing, y_spacing = 0.05, 0.015
    lefts = [0.05 + i * (width + x_spacing) for i in range(n_cols)]
    bottom = 0.05 + (2) * (height + y_spacing)  # top to bottom

    fig = plt.figure(figsize=(10, 20))
    axes_grid = [None for _ in range(n_cols)]

    # Loop over data rows and columns
    for col in range(n_cols):
        left = lefts[col]
        ax = fig.add_axes([left, bottom, width, height])
        axes_grid[col] = ax

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        extent = (0.5, n_som_xy[0] + 0.5, 0.5, n_som_xy[1] + 0.5)
        im = ax.imshow(data_[col].T, extent=extent, origin='lower',
                    cmap=cmaps[col], vmin=vmins[col], vmax=vmaxs[col])

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
    
    #plt.show()

    if filename!=None:
        fig.savefig(filename, dpi=500, bbox_inches='tight')

    return count_df_all

def plot_som_valid(full_df, valid_df, map, size, filename=None):
    '''
    For qualitative SOM optimization, plots the three-column figure 
    where the N column shows the density of the full sample, but the
    redshift and sSFR columns show only the validation data. This way
    only the redshift labels of the validation set are used (similarly
    to optimizing the UMAP embedding using only the validation data).

    Parameters
    ----------
    phot_df: DataFrame containing full photometric catalog, must contain some form of 'SOM-1' and 'SOM-2' coords
    spec_df: DataFrame containing spec-z catalog, must contain some form of 'SOM-1' and 'SOM-2' coords
    map: SOM lead string, e.g., SOM, SOM8, SOM75, etc.
    size: tuple of integers describing SOM size

    Returns
    ----------
    3-column SOM plot
    counts_all, counts_lp, counts_s
    '''
    #CODE
    df1 = full_df.copy()
    df2 = valid_df.copy()
    n_som_xy = size
    coords = [map+'-1', map+'-2']

    # 1. Count of entries at each (X, Y) grid point
    count_df_all = df1.groupby(coords).size().reset_index(name='count')
    cell_counts = count_df_all.pivot(index=coords[0], columns=coords[1], values='count').fillna(0).to_numpy()

    # 2. Median of 'z' at each (X, Y)
    z_median_df = df2.groupby(coords)['lp_photoz'].median().reset_index()
    z_median_cell_v = z_median_df.pivot(index=coords[0], columns=coords[1], values='lp_photoz').to_numpy()

    # 3. Median of 's' at each (X, Y)
    s_median_df = df2.groupby(coords)['lp_sSFR_best'].median().reset_index()
    ssfr_median_cell_v = s_median_df.pivot(index=coords[0], columns=coords[1], values='lp_sSFR_best').to_numpy()

    data_ = (np.log10(cell_counts), z_median_cell_v, ssfr_median_cell_v)


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
    n_cols = 3
    width, height = 0.25, 0.25
    x_spacing, y_spacing = 0.05, 0.015
    lefts = [0.05 + i * (width + x_spacing) for i in range(n_cols)]
    bottom = 0.05 + (2) * (height + y_spacing)  # top to bottom

    fig = plt.figure(figsize=(10, 20))
    axes_grid = [None for _ in range(n_cols)]

    # Loop over data rows and columns
    for col in range(n_cols):
        left = lefts[col]
        ax = fig.add_axes([left, bottom, width, height])
        axes_grid[col] = ax

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        extent = (0.5, n_som_xy[0] + 0.5, 0.5, n_som_xy[1] + 0.5)
        im = ax.imshow(data_[col].T, extent=extent, origin='lower',
                    cmap=cmaps[col], vmin=vmins[col], vmax=vmaxs[col])

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
    
    #plt.show()

    if filename!=None:
        fig.savefig(filename, dpi=500, bbox_inches='tight')

    return count_df_all

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
    elif color in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:
        sc = ax.scatter(df['Z1'], df['X1'], -df['Y1'], c=color, s=7, alpha=0.5)
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
        #sc = plot_data(df, color, cbar=cbar)
        if color2!=None:
            #plot_data(df2, color2, cbar=None)
            sc = plot_data_grays(dfc=df, dfg=df2, color=color, color2=color2, cbar=True)
        else:
            sc = plot_data(df, color, cbar=cbar)
        
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

def plot_data_grays(dfc, dfg, color='lp_photoz', color2='0.5', cbar=True):
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

    ax.scatter(dfg['Z1'], dfg['X1'], -dfg['Y1'], c=color2, s=0.5, alpha=0.5)

    # Create scatter plot
    if color == 'lp_photoz':
        sc = ax.scatter(dfc['Z1'], dfc['X1'], -dfc['Y1'], c=dfc[color], s=2., alpha=1.0, cmap=get_cmap('cet_CET_R1'), vmin=0, vmax=5)
    elif color == 'Match_specz':
        sc = ax.scatter(dfc['Z1'], dfc['X1'], -dfc['Y1'], c=dfc[color], s=1, alpha=0.7, cmap=get_cmap('cet_CET_R1'), vmin=0, vmax=5)
    elif color == 'specz':
        sc = ax.scatter(dfc['Z1'], dfc['X1'], -dfc['Y1'], c=dfc[color], s=2., alpha=1.0, cmap=get_cmap('cet_CET_R1'), vmin=0, vmax=5)
    elif color == 'lp_sSFR_best':
        sc = ax.scatter(dfc['Z1'], dfc['X1'], -dfc['Y1'], c=dfc[color], s=1., alpha=0.7, cmap=get_cmap('cet_CET_D1A_r'), vmin=-12, vmax=-8)
    elif color == 'black':
        sc = ax.scatter(dfc['Z1'], dfc['X1'], -dfc['Y1'], c='k', s=7, alpha=1.)
    else:
        sc = ax.scatter(dfc['Z1'], dfc['X1'], -dfc['Y1'], c=dfc[color], s=1, alpha=0.7, cmap=get_cmap('cet_CET_R1'))

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


def z_binned_som(fullcat_lp, fullcat_spec, som_test_lp, som_test_spec, i_vals='TEST_Z', t_vals='lp_photoz', bin_min=0.0, bin_max=3.0, bin_size=0.2, umap_test_lp=[], umap_test_spec=[], filename=None):
    '''This function creates a multi-panel SOM plot, separating the matched (trained cells) and cell-rematched statistics.

    Parameters
    ----------
    fullcat_lp: pandas DataFrame
        DataFrame containing the full catalog of test objects, LePHARE-trained case
    fullcat_spec: pandas DataFrame
        DataFrame containing the full catalog of test objects, spec-z-trained case
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
    som_lp_stats = binned_stats(fullcat=fullcat_lp, testcat=som_test_lp, iv=i_vals, tv=t_vals, 
                                bin_var=t_vals, bin_min=bin_min, bin_max=bin_max, bin_size=bin_size)
    som_spec_stats = binned_stats(fullcat=fullcat_spec, testcat=som_test_spec, iv=i_vals, tv=t_vals, 
                                  bin_var=t_vals, bin_min=bin_min, bin_max=bin_max, bin_size=bin_size)
    # create the figure
    fig, axs = plt.subplots(3, 2, figsize=(16,12))

    bin_centers = som_lp_stats['bin_centers']

    fout_ymax = 1.05*np.max([np.max(som_lp_stats['outliers_p']), np.max(som_spec_stats['m_outliers']), 
                             np.max(som_spec_stats['outliers_p']), np.max(som_lp_stats['rm_outliers']), 
                             np.max(som_spec_stats['rm_outliers']), np.max(som_lp_stats['rm_only_outliers']), 
                             np.max(som_spec_stats['rm_only_outliers'])])
    nmad_ymax = 1.05*np.max([np.max(som_lp_stats['rm_nmads']), np.max(som_spec_stats['rm_nmads']), 
                             np.max(som_lp_stats['m_nmads']), np.max(som_spec_stats['m_nmads']),
                             np.max(som_lp_stats['rm_only_nmads']), np.max(som_spec_stats['rm_only_nmads'])])
    bias_ymax = 1.05*np.max([np.max(som_lp_stats['rm_biases']), np.max(som_spec_stats['rm_biases']), 
                             np.max(som_lp_stats['m_biases']), np.max(som_spec_stats['m_biases']),
                             np.max(som_lp_stats['rm_only_biases']), np.max(som_spec_stats['rm_only_biases'])])

    c1, c2, c3, c4 = '0.3', 'red', '0.5', 'g'
    # UPPER LEFT
    axs[0,0].plot(bin_centers, som_lp_stats['outliers_p'], c=c1, label='SOM (LePHARE-trained, all cells)', ls='--')
    axs[0,0].plot(bin_centers, som_lp_stats['m_outliers'], c=c3, alpha=0.7, label='SOM (LePHARE-trained, cells with training redshifts)', ls='--')
    axs[0,0].plot(bin_centers, som_lp_stats['rm_outliers'], c=c2, alpha=0.7, label='SOM (LePHARE-trained, all cells incl. re-matching)', ls='--')
    axs[0,0].plot(bin_centers, som_lp_stats['rm_only_outliers'], c=c4, alpha=0.7, label='SOM (LePHARE-trained, re-matched objects only)', ls='--')
    axs[0,0].fill_between(bin_centers, som_lp_stats['outliers_p'], y2=som_lp_stats['m_outliers'], alpha=0.15, color='0.4')
    axs[0,0].scatter(bin_centers, som_lp_stats['outliers_p'], c=c1, s=20)
    axs[0,0].scatter(bin_centers, som_lp_stats['m_outliers'], c=c3, alpha=0.7, s=20)
    axs[0,0].scatter(bin_centers, som_lp_stats['rm_outliers'], c=c2, alpha=0.7, s=20)
    axs[0,0].scatter(bin_centers, som_lp_stats['rm_only_outliers'], c=c4, alpha=0.7, s=20)
    axs[0,0].set_ylabel(r'$f_\mathrm{outlier}$', fontsize=25)
    axs[0,0].tick_params(axis='y', labelsize=18)
    axs[0,0].tick_params(axis='x', labelsize=18)
    axs[0,0].set_ylim(0,fout_ymax)
    axs[0,0].set_xlim(bin_min,None)
    # MIDDLE LEFT
    axs[1,0].plot(bin_centers, som_lp_stats['m_nmads'], c=c3, alpha=0.7, ls='--')
    axs[1,0].scatter(bin_centers, som_lp_stats['m_nmads'], c=c3, alpha=0.7, s=20)
    axs[1,0].plot(bin_centers, som_lp_stats['rm_nmads'], c=c2, alpha=0.7, ls='--')
    axs[1,0].scatter(bin_centers, som_lp_stats['rm_nmads'], c=c2, alpha=0.7, s=20)
    axs[1,0].plot(bin_centers, som_lp_stats['rm_only_nmads'], c=c4, alpha=0.7, ls='--')
    axs[1,0].scatter(bin_centers, som_lp_stats['rm_only_nmads'], c=c4, alpha=0.7, s=20)
    axs[1,0].set_ylabel(r'$\sigma_\mathrm{NMAD}$', fontsize=25)
    axs[1,0].tick_params(axis='y', labelsize=18)
    axs[1,0].tick_params(axis='x', labelsize=18)
    axs[1,0].set_ylim(0,nmad_ymax)
    axs[1,0].set_xlim(bin_min,None)
    # BOTTOM LEFT
    axs[2,0].plot(bin_centers, som_lp_stats['m_biases'], c=c3, alpha=0.7, ls='--')
    axs[2,0].scatter(bin_centers, som_lp_stats['m_biases'], c=c3, alpha=0.7, s=20)
    axs[2,0].plot(bin_centers, som_lp_stats['rm_biases'], c=c2, alpha=0.7, ls='--')
    axs[2,0].scatter(bin_centers, som_lp_stats['rm_biases'], c=c2, alpha=0.7, s=20)
    axs[2,0].plot(bin_centers, som_lp_stats['rm_only_biases'], c=c4, alpha=0.7, ls='--')
    axs[2,0].scatter(bin_centers, som_lp_stats['rm_only_biases'], c=c4, alpha=0.7, s=20)
    axs[2,0].axhline(y = 0.0, color = 'k', linestyle = ':', lw=1.3, alpha=0.8)
    axs[2,0].set_ylim(-bias_ymax,bias_ymax)
    axs[2,0].set_xlim(bin_min,None)
    axs[2,0].set_ylabel('Bias ('+r'$\langle \frac{\Delta z}{1+z} \rangle$'+')', fontsize=25)
    axs[2,0].tick_params(axis='y', labelsize=18)
    axs[2,0].tick_params(axis='x', labelsize=18)
    axs[2,0].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)
    # UPPER RIGHT
    axs[0,1].plot(bin_centers, som_spec_stats['outliers_p'], c=c1, label='SOM (spec-'+r'$z$'+'-trained, all cells)')
    axs[0,1].plot(bin_centers, som_spec_stats['m_outliers'], c=c3, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, cells with training redshifts)')
    axs[0,1].plot(bin_centers, som_spec_stats['rm_outliers'], c=c2, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, all cells incl. re-matching)')
    axs[0,1].plot(bin_centers, som_spec_stats['rm_only_outliers'], c=c4, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, re-matched objects only)')
    axs[0,1].fill_between(bin_centers, som_spec_stats['outliers_p'], y2=som_spec_stats['m_outliers'], alpha=0.15, color='0.4')
    axs[0,1].scatter(bin_centers, som_spec_stats['outliers_p'], c=c1, s=20)
    axs[0,1].scatter(bin_centers, som_spec_stats['m_outliers'], c=c3, alpha=0.7, s=20)
    axs[0,1].scatter(bin_centers, som_spec_stats['rm_outliers'], c=c2, alpha=0.7, s=20)
    axs[0,1].scatter(bin_centers, som_spec_stats['rm_only_outliers'], c=c4, alpha=0.7, s=20)   
    axs[0,1].tick_params(axis='y', labelsize=18)
    axs[0,1].tick_params(axis='x', labelsize=18)
    axs[0,1].set_ylim(0,fout_ymax)
    axs[0,1].set_xlim(bin_min,None)
    # MIDDLE RIGHT
    axs[1,1].plot(bin_centers, som_spec_stats['m_nmads'], c=c3, alpha=0.7)
    axs[1,1].scatter(bin_centers, som_spec_stats['m_nmads'], c=c3, alpha=0.7, s=20)
    axs[1,1].plot(bin_centers, som_spec_stats['rm_nmads'], c=c2)
    axs[1,1].scatter(bin_centers, som_spec_stats['rm_nmads'], c=c2, alpha=0.7, s=20)
    axs[1,1].plot(bin_centers, som_spec_stats['rm_only_nmads'], c=c4)
    axs[1,1].scatter(bin_centers, som_spec_stats['rm_only_nmads'], c=c4, alpha=0.7, s=20)    
    axs[1,1].tick_params(axis='y', labelsize=18)
    axs[1,1].tick_params(axis='x', labelsize=18)
    axs[1,1].set_ylim(0,nmad_ymax)
    axs[1,1].set_xlim(bin_min,None)
    # BOTTOM RIGHT
    axs[2,1].plot(bin_centers, som_spec_stats['m_biases'], c=c3, alpha=0.7)
    axs[2,1].scatter(bin_centers, som_spec_stats['m_biases'], c=c3, alpha=0.7, s=20)
    axs[2,1].plot(bin_centers, som_spec_stats['rm_biases'], c=c2)
    axs[2,1].scatter(bin_centers, som_spec_stats['rm_biases'], c=c2, alpha=0.7, s=20)
    axs[2,1].plot(bin_centers, som_spec_stats['rm_only_biases'], c=c4)
    axs[2,1].scatter(bin_centers, som_spec_stats['rm_only_biases'], c=c4, alpha=0.7, s=20)
    axs[2,1].axhline(y = 0.0, color = 'k', linestyle =':', lw=1.3, alpha=0.8)
    axs[2,1].set_ylim(-bias_ymax,bias_ymax)
    axs[2,1].set_xlim(bin_min,None)
    axs[2,1].tick_params(axis='y', labelsize=18)
    axs[2,1].tick_params(axis='x', labelsize=18)
    axs[2,1].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)

    if len(umap_test_lp)!=0:
        umap_lp_stats = binned_stats_old(fullcat=umap_test_lp, testcat=umap_test_lp, iv=i_vals, tv=t_vals, 
                                 bin_var=t_vals, bin_min=bin_min, bin_max=bin_max, bin_size=bin_size)
        umap_spec_stats = binned_stats_old(fullcat=umap_test_spec, testcat=umap_test_spec, iv=i_vals, tv=t_vals, 
                                   bin_var=t_vals, bin_min=bin_min, bin_max=bin_max, bin_size=bin_size)
        axs[0,0].plot(bin_centers, umap_lp_stats['outliers'], c='blue', label='UMAP (LePHARE-trained, re-matched only)', ls='--')
        axs[0,0].scatter(bin_centers, umap_lp_stats['outliers'], c='blue', s=20)
        axs[1,0].plot(bin_centers, umap_lp_stats['nmads'], c='blue', ls='--')
        axs[1,0].scatter(bin_centers, umap_lp_stats['nmads'], c='blue', s=20)
        axs[2,0].plot(bin_centers, umap_lp_stats['biases'], c='blue', ls='--')
        axs[2,0].scatter(bin_centers, umap_lp_stats['biases'], c='blue', s=20)
        axs[0,1].plot(bin_centers, umap_lp_stats['outliers'], c='blue', ls='--')
        axs[0,1].scatter(bin_centers, umap_lp_stats['outliers'], c='blue', s=20)
        axs[0,1].plot(bin_centers, umap_spec_stats['outliers'], c='blue', label='UMAP (spec-'+r'$z$'+'-trained, re-matched only)')
        axs[0,1].scatter(bin_centers, umap_spec_stats['outliers'], c='blue', s=20)
        axs[1,1].plot(bin_centers, umap_lp_stats['nmads'], c='blue', ls='--')
        axs[1,1].scatter(bin_centers, umap_lp_stats['nmads'], c='blue', s=20)
        axs[1,1].plot(bin_centers, umap_spec_stats['nmads'], c='blue')
        axs[1,1].scatter(bin_centers, umap_spec_stats['nmads'], c='blue', s=20)
        axs[2,1].plot(bin_centers, umap_lp_stats['biases'], c='blue', ls='--')
        axs[2,1].scatter(bin_centers, umap_lp_stats['biases'], c='blue', s=20)
        axs[2,1].plot(bin_centers, umap_spec_stats['biases'], c='blue')
        axs[2,1].scatter(bin_centers, umap_spec_stats['biases'], c='blue', s=20)

    fig.subplots_adjust(top=0.95)
    fig.legend(bbox_to_anchor=(0.03, 1.02, 1., .102), loc='center',
                        ncols=2, borderaxespad=0., fontsize=17)
    fig.tight_layout()
    #fig.suptitle('SOM vs UMAP: Statistics by Redshift', fontsize=14);
    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')


# create a Figure with performance statistics binned by redshift, corresponding to Figure 5 in our paper
def z_binned_stats(fullcat_lp, fullcat_spec, umap_test_lp, umap_test_spec, som_test_lp, som_test_spec, umap_errs, som_rm_errs, som_nrm_errs, i_vals='TEST_Z', t_vals='lp_photoz', bin_min=0.0, bin_max=3.0, bin_size=0.2, filename=None):
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
    umap_errs: dictionary
        dictionary containing the redshift-binned uncertainties for UMAP, with the keys
        'outlier', 'nmad', 'bias' containing the uncertainties on the specified metric
    som_errs: dictionary
        dictionary containing the redshift-binned uncertainties for SOM, with the keys
        'outlier', 'nmad', 'bias' containing the uncertainties on the specified metric
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
    umap_lp_stats = binned_stats_old(fullcat=fullcat_lp, testcat=umap_test_lp, iv=i_vals, tv=t_vals, 
                                 bin_var=t_vals, bin_min=bin_min, bin_max=bin_max, bin_size=bin_size)
    som_lp_stats = binned_stats(fullcat=fullcat_lp, testcat=som_test_lp, iv=i_vals, tv=t_vals, 
                                bin_var=t_vals, bin_min=bin_min, bin_max=bin_max, bin_size=bin_size)
    # calculate z-binned stats, spec-z-trained case
    umap_spec_stats = binned_stats_old(fullcat=fullcat_spec, testcat=umap_test_spec, iv=i_vals, tv=t_vals, 
                                   bin_var=t_vals, bin_min=bin_min, bin_max=bin_max, bin_size=bin_size)
    som_spec_stats = binned_stats(fullcat=fullcat_spec, testcat=som_test_spec, iv=i_vals, tv=t_vals, 
                                  bin_var=t_vals, bin_min=bin_min, bin_max=bin_max, bin_size=bin_size)
    # create the figure
    fig, axs = plt.subplots(3, 2, figsize=(16,12))

    bin_centers = umap_lp_stats['bin_centers']

    fout_ymax = 0.95*np.max(#[np.max(som_lp_stats['outliers_p']), 
                             [np.max(umap_lp_stats['outliers']), np.max(som_spec_stats['m_outliers']), 
                             np.max(umap_spec_stats['outliers'])#, np.max(som_spec_stats['outliers_p'])
                             ]) + np.max([np.max(som_rm_errs['outliers']), np.max(som_rm_errs['outliers']), np.max(umap_errs['outliers'])])
    nmad_ymax = 1.01*np.max([np.max(som_lp_stats['rm_nmads']), np.max(umap_lp_stats['nmads']), np.max(som_spec_stats['rm_nmads']), 
                             np.max(umap_spec_stats['nmads']), np.max(som_lp_stats['m_nmads']), 
                             np.max(som_spec_stats['m_nmads'])])+np.max([np.max(som_rm_errs['nmads']), np.max(som_nrm_errs['nmads']), np.max(umap_errs['nmads'])])
    bias_ymax = 1.01*np.max([np.max(som_lp_stats['rm_biases']), np.max(umap_lp_stats['biases']), np.max(som_spec_stats['rm_biases']), 
                             np.max(umap_spec_stats['biases']), np.max(som_lp_stats['m_biases']), 
                             np.max(som_spec_stats['m_biases'])])+np.max([np.max(som_rm_errs['biases']), np.max(som_nrm_errs['biases']), np.max(umap_errs['biases'])])

    c1, c2, c3 = '0.3', 'red', '0.5'
    umap_lp_alpha=0.4
    # UPPER LEFT
    metric='outliers'
    #axs[0,0].plot(bin_centers, som_lp_stats[metric+'_p'], c=c1, label='SOM (LePHARE-trained, all cells)', ls='--')
    axs[0,0].plot(bin_centers, som_lp_stats['m_'+metric], c=c3, alpha=0.7, label='SOM (LePHARE-trained, cells with training redshifts)', ls='--')
    axs[0,0].plot(bin_centers, som_lp_stats['rm_'+metric], c=c2, alpha=0.7, label='SOM (LePHARE-trained, all cells incl. re-matching)', ls='--')
    #axs[0,0].fill_between(bin_centers, som_lp_stats[metric+'_p'], y2=som_lp_stats['m_'+metric], alpha=0.15, color='0.4') 
    #axs[0,0].errorbar(bin_centers, som_lp_stats[metric+'_p'], yerr=som_errs[metric], fmt='o', capsize=3,
    #                lw=1, ms=4, c=c1)
    axs[0,0].errorbar(bin_centers, som_lp_stats['m_'+metric], yerr=som_nrm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[0,0].errorbar(bin_centers, som_lp_stats['rm_'+metric], yerr=som_rm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[0,0].plot(bin_centers, umap_lp_stats[metric], c='blue', label='UMAP (LePHARE-trained)', ls='--')
    axs[0,0].errorbar(bin_centers, umap_lp_stats[metric], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[0,0].set_ylabel(r'$f_\mathrm{outlier}$', fontsize=25)
    axs[0,0].tick_params(axis='y', labelsize=18)
    axs[0,0].tick_params(axis='x', labelsize=18)
    axs[0,0].set_ylim(0,fout_ymax)
    axs[0,0].set_xlim(bin_min,None)
    # MIDDLE LEFT
    metric='nmads'
    axs[1,0].plot(bin_centers, som_lp_stats['m_'+metric], c=c3, alpha=0.7, ls='--')
    axs[1,0].plot(bin_centers, som_lp_stats['rm_'+metric], c=c2, alpha=0.7, ls='--')
    axs[1,0].errorbar(bin_centers, som_lp_stats['m_'+metric], yerr=som_nrm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[1,0].errorbar(bin_centers, som_lp_stats['rm_'+metric], yerr=som_rm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[1,0].plot(bin_centers, umap_lp_stats[metric], c='blue', ls='--')
    axs[1,0].errorbar(bin_centers, umap_lp_stats[metric], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[1,0].set_ylabel(r'$\sigma_\mathrm{NMAD}$', fontsize=25)
    axs[1,0].tick_params(axis='y', labelsize=18)
    axs[1,0].tick_params(axis='x', labelsize=18)
    axs[1,0].set_ylim(0,nmad_ymax)
    axs[1,0].set_xlim(bin_min,None)
    # BOTTOM LEFT
    metric='biases'
    axs[2,0].plot(bin_centers, som_lp_stats['m_'+metric], c=c3, alpha=0.7, ls='--')
    axs[2,0].plot(bin_centers, som_lp_stats['rm_'+metric], c=c2, alpha=0.7, ls='--')
    axs[2,0].errorbar(bin_centers, som_lp_stats['m_'+metric], yerr=som_nrm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[2,0].errorbar(bin_centers, som_lp_stats['rm_'+metric], yerr=som_rm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[2,0].plot(bin_centers, umap_lp_stats[metric], c='blue', ls='--')
    axs[2,0].errorbar(bin_centers, umap_lp_stats[metric], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[2,0].axhline(y = 0.0, color = 'k', linestyle = ':', lw=1.3, alpha=0.8)
    axs[2,0].set_ylim(-bias_ymax,bias_ymax)
    axs[2,0].set_xlim(bin_min,None)
    axs[2,0].set_ylabel('Bias ('+r'$\langle \frac{\Delta z}{1+z} \rangle$'+')', fontsize=25)
    axs[2,0].tick_params(axis='y', labelsize=18)
    axs[2,0].tick_params(axis='x', labelsize=18)
    axs[2,0].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)
    # UPPER RIGHT
    metric='outliers'
    axs[0,1].plot(bin_centers, umap_lp_stats[metric], c='blue', ls='--', alpha=umap_lp_alpha)
    axs[0,1].errorbar(bin_centers, umap_lp_stats[metric], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue', alpha=umap_lp_alpha)
    #axs[0,1].plot(bin_centers, som_spec_stats[metric+'_p'], c=c1, label='SOM (spec-'+r'$z$'+'-trained, all cells)', ls='-')
    axs[0,1].plot(bin_centers, som_spec_stats['m_'+metric], c=c3, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, cells with training redshifts)', ls='-')
    axs[0,1].plot(bin_centers, som_spec_stats['rm_'+metric], c=c2, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, all cells incl. re-matching)', ls='-')
    #axs[0,1].fill_between(bin_centers, som_spec_stats[metric+'_p'], y2=som_spec_stats['m_'+metric], alpha=0.15, color='0.4')
    #axs[0,1].errorbar(bin_centers, som_spec_stats[metric+'_p'], yerr=som_errs[metric], fmt='o', capsize=3,
    #                lw=1, ms=4, c=c1)
    axs[0,1].errorbar(bin_centers, som_spec_stats['m_'+metric], yerr=som_nrm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[0,1].errorbar(bin_centers, som_spec_stats['rm_'+metric], yerr=som_rm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[0,1].plot(bin_centers, umap_spec_stats[metric], c='blue', label='UMAP (spec-'+r'$z$'+'-trained)', ls='-')
    axs[0,1].errorbar(bin_centers, umap_spec_stats[metric], yerr=umap_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[0,1].tick_params(axis='y', labelsize=18)
    axs[0,1].tick_params(axis='x', labelsize=18)
    axs[0,1].set_ylim(0,fout_ymax)
    axs[0,1].set_xlim(bin_min,None)
    # MIDDLE RIGHT
    metric='nmads'
    axs[1,1].plot(bin_centers, som_spec_stats['m_'+metric], c=c3, alpha=0.7, ls='-')
    axs[1,1].plot(bin_centers, som_spec_stats['rm_'+metric], c=c2, alpha=0.7, ls='-')
    axs[1,1].errorbar(bin_centers, som_spec_stats['m_'+metric], yerr=som_nrm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[1,1].errorbar(bin_centers, som_spec_stats['rm_'+metric], yerr=som_rm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[1,1].plot(bin_centers, umap_spec_stats[metric], c='blue', ls='-')
    axs[1,1].errorbar(bin_centers, umap_spec_stats[metric], yerr=umap_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[1,1].plot(bin_centers, umap_lp_stats[metric], c='blue', ls='--', alpha=umap_lp_alpha)
    axs[1,1].errorbar(bin_centers, umap_lp_stats[metric], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue', alpha=umap_lp_alpha)
    axs[1,1].tick_params(axis='y', labelsize=18)
    axs[1,1].tick_params(axis='x', labelsize=18)
    axs[1,1].set_ylim(0,nmad_ymax)
    axs[1,1].set_xlim(bin_min,None)
    # BOTTOM RIGHT
    metric='biases'
    axs[2,1].plot(bin_centers, som_spec_stats['m_'+metric], c=c3, alpha=0.7, ls='-')
    axs[2,1].plot(bin_centers, som_spec_stats['rm_'+metric], c=c2, alpha=0.7, ls='-')
    axs[2,1].errorbar(bin_centers, som_spec_stats['m_'+metric], yerr=som_nrm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[2,1].errorbar(bin_centers, som_spec_stats['rm_'+metric], yerr=som_rm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[2,1].plot(bin_centers, umap_spec_stats[metric], c='blue', ls='-')
    axs[2,1].errorbar(bin_centers, umap_spec_stats[metric], yerr=umap_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[2,1].plot(bin_centers, umap_lp_stats[metric], c='blue', ls='--', alpha=umap_lp_alpha)
    axs[2,1].errorbar(bin_centers, umap_lp_stats[metric], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue', alpha=umap_lp_alpha)
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

# create a Figure with performance statistics binned by redshift, corresponding to Figure 5 in our paper
def z_binned_stats2(som_nrm_s, som_rm_s, umap_s, som_nrm_lp, som_rm_lp, umap_lp, umap_errs, som_rm_errs, som_nrm_errs, i_vals='TEST_Z', t_vals='lp_photoz', bin_min=0.0, bin_max=3.0, bin_size=0.2, filename=None):
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
    umap_errs: dictionary
        dictionary containing the redshift-binned uncertainties for UMAP, with the keys
        'outlier', 'nmad', 'bias' containing the uncertainties on the specified metric
    som_errs: dictionary
        dictionary containing the redshift-binned uncertainties for SOM, with the keys
        'outlier', 'nmad', 'bias' containing the uncertainties on the specified metric
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

    # create the figure
    fig, axs = plt.subplots(3, 2, figsize=(16,12))

    bin_centers = umap_s['bin_centers']

    metric='outliers'
    fout_ymax = 0.86*np.max([np.max(umap_s[metric+'_mean']), np.max(umap_lp[metric+'_mean']), 
                            np.max(som_rm_s[metric+'_mean']), np.max(som_rm_lp[metric+'_mean']),
                            np.max(som_nrm_s[metric+'_mean']), np.max(som_nrm_lp[metric+'_mean'])]) + np.max([np.max(som_nrm_errs[metric]), 
                            np.max(som_rm_errs[metric]), np.max(umap_errs[metric]), np.max(som_nrm_errs[metric+'_lp']), 
                            np.max(som_rm_errs[metric+'_lp']), np.max(umap_errs[metric+'_lp'])])
    metric='nmads'
    nmad_ymax = 1.01*np.max([np.max(umap_s[metric+'_mean']), np.max(umap_lp[metric+'_mean']), 
                            np.max(som_rm_s[metric+'_mean']), np.max(som_rm_lp[metric+'_mean']),
                            np.max(som_nrm_s[metric+'_mean']), np.max(som_nrm_lp[metric+'_mean'])]) + np.max([np.max(som_nrm_errs[metric]), 
                            np.max(som_rm_errs[metric]), np.max(umap_errs[metric]), np.max(som_nrm_errs[metric+'_lp']), 
                            np.max(som_rm_errs[metric+'_lp']), np.max(umap_errs[metric+'_lp'])])
    metric='biases'
    bias_ymax = 1.01*np.max([np.max(umap_s[metric+'_mean']), np.max(umap_lp[metric+'_mean']), 
                            np.max(som_rm_s[metric+'_mean']), np.max(som_rm_lp[metric+'_mean']),
                            np.max(som_nrm_s[metric+'_mean']), np.max(som_nrm_lp[metric+'_mean'])]) + np.max([np.max(som_nrm_errs[metric]), 
                            np.max(som_rm_errs[metric]), np.max(umap_errs[metric]), np.max(som_nrm_errs[metric+'_lp']), 
                            np.max(som_rm_errs[metric+'_lp']), np.max(umap_errs[metric+'_lp'])])

    c1, c2, c3 = '0.3', 'red', '0.5'
    umap_lp_alpha=0.4

    #ticks
    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=18, length=6, width=1.5, 
                        bottom=True,top=False,left=True, right=False,
                        direction='out')

    # UPPER LEFT
    metric='outliers'
    axs[0,0].plot(bin_centers, som_nrm_lp[metric+'_mean'], c=c3, alpha=0.7, label='SOM (LePHARE-trained, cells with training redshifts)', ls='--')
    axs[0,0].plot(bin_centers, som_rm_lp[metric+'_mean'], c=c2, alpha=0.7, label='SOM (LePHARE-trained, all cells with re-matching)', ls='--')
    axs[0,0].errorbar(bin_centers, som_nrm_lp[metric+'_mean'], yerr=som_nrm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[0,0].errorbar(bin_centers, som_rm_lp[metric+'_mean'], yerr=som_rm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[0,0].plot(bin_centers, umap_lp[metric+'_mean'], c='blue', label='UMAP (LePHARE-trained)', ls='--')
    axs[0,0].errorbar(bin_centers, umap_lp[metric+'_mean'], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[0,0].set_ylabel(r'$f_\mathrm{outlier}$', fontsize=25)
    #axs[0,0].tick_params(axis='both', labelsize=18, length=6, width=1.5, 
    #                     bottom=True,top=False,left=True, right=False,
    #                     direction='out')
    #axs[0,0].tick_params(axis='x', labelsize=18, length=10, width=10)
    axs[0,0].set_ylim(0,fout_ymax)
    axs[0,0].set_xlim(bin_min,None)
    # MIDDLE LEFT
    metric='nmads'
    axs[1,0].plot(bin_centers, som_nrm_lp[metric+'_mean'], c=c3, alpha=0.7, ls='--')
    axs[1,0].plot(bin_centers, som_rm_lp[metric+'_mean'], c=c2, alpha=0.7, ls='--')
    axs[1,0].errorbar(bin_centers, som_nrm_lp[metric+'_mean'], yerr=som_nrm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[1,0].errorbar(bin_centers, som_rm_lp[metric+'_mean'], yerr=som_rm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[1,0].plot(bin_centers, umap_lp[metric+'_mean'], c='blue', ls='--')
    axs[1,0].errorbar(bin_centers, umap_lp[metric+'_mean'], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[1,0].set_ylabel(r'$\sigma_\mathrm{NMAD}$', fontsize=25)
    #axs[1,0].tick_params(axis='y', labelsize=18)
    #axs[1,0].tick_params(axis='x', labelsize=18)
    axs[1,0].set_ylim(0,nmad_ymax)
    axs[1,0].set_xlim(bin_min,None)
    # BOTTOM LEFT
    metric='biases'
    axs[2,0].plot(bin_centers, som_nrm_lp[metric+'_mean'], c=c3, alpha=0.7, ls='--')
    axs[2,0].plot(bin_centers, som_rm_lp[metric+'_mean'], c=c2, alpha=0.7, ls='--')
    axs[2,0].errorbar(bin_centers, som_nrm_lp[metric+'_mean'], yerr=som_nrm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[2,0].errorbar(bin_centers, som_rm_lp[metric+'_mean'], yerr=som_rm_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[2,0].plot(bin_centers, umap_lp[metric+'_mean'], c='blue', ls='--')
    axs[2,0].errorbar(bin_centers, umap_lp[metric+'_mean'], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[2,0].axhline(y = 0.0, color = 'k', linestyle = ':', lw=1.3, alpha=0.8)
    axs[2,0].set_ylim(-bias_ymax,bias_ymax)
    axs[2,0].set_xlim(bin_min,None)
    axs[2,0].set_ylabel('Bias ('+r'$\langle \frac{\Delta z}{1+z} \rangle$'+')', fontsize=25)
    #axs[2,0].tick_params(axis='y', labelsize=18)
    #axs[2,0].tick_params(axis='x', labelsize=18)
    axs[2,0].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)
    # UPPER RIGHT
    metric='outliers'
    axs[0,1].plot(bin_centers, umap_lp[metric+'_mean'], c='blue', ls='--', alpha=umap_lp_alpha)
    axs[0,1].errorbar(bin_centers, umap_lp[metric+'_mean'], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue', alpha=umap_lp_alpha)
    axs[0,1].plot(bin_centers, som_nrm_s[metric+'_mean'], c=c3, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, cells with training redshifts)', ls='-')
    axs[0,1].plot(bin_centers, som_rm_s[metric+'_mean'], c=c2, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, all cells with re-matching)', ls='-')
    axs[0,1].errorbar(bin_centers, som_nrm_s[metric+'_mean'], yerr=som_nrm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[0,1].errorbar(bin_centers, som_rm_s[metric+'_mean'], yerr=som_rm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[0,1].plot(bin_centers, umap_s[metric+'_mean'], c='blue', label='UMAP (spec-'+r'$z$'+'-trained)', ls='-')
    axs[0,1].errorbar(bin_centers, umap_s[metric+'_mean'], yerr=umap_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    #axs[0,1].tick_params(axis='y', labelsize=18)
    #axs[0,1].tick_params(axis='x', labelsize=18)
    axs[0,1].set_ylim(0,fout_ymax)
    axs[0,1].set_xlim(bin_min,None)
    # MIDDLE RIGHT
    metric='nmads'
    axs[1,1].plot(bin_centers, som_nrm_s[metric+'_mean'], c=c3, alpha=0.7, ls='-')
    axs[1,1].plot(bin_centers, som_rm_s[metric+'_mean'], c=c2, alpha=0.7, ls='-')
    axs[1,1].errorbar(bin_centers, som_nrm_s[metric+'_mean'], yerr=som_nrm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[1,1].errorbar(bin_centers, som_rm_s[metric+'_mean'], yerr=som_rm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[1,1].plot(bin_centers, umap_s[metric+'_mean'], c='blue', ls='-')
    axs[1,1].errorbar(bin_centers, umap_s[metric+'_mean'], yerr=umap_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[1,1].plot(bin_centers, umap_lp[metric+'_mean'], c='blue', ls='--', alpha=umap_lp_alpha)
    axs[1,1].errorbar(bin_centers, umap_lp[metric+'_mean'], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue', alpha=umap_lp_alpha)
    #axs[1,1].tick_params(axis='y', labelsize=18)
    #axs[1,1].tick_params(axis='x', labelsize=18)
    axs[1,1].set_ylim(0,nmad_ymax)
    axs[1,1].set_xlim(bin_min,None)
    # BOTTOM RIGHT
    metric='biases'
    axs[2,1].plot(bin_centers, som_nrm_s[metric+'_mean'], c=c3, alpha=0.7, ls='-')
    axs[2,1].plot(bin_centers, som_rm_s[metric+'_mean'], c=c2, alpha=0.7, ls='-')
    axs[2,1].errorbar(bin_centers, som_nrm_s[metric+'_mean'], yerr=som_nrm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[2,1].errorbar(bin_centers, som_rm_s[metric+'_mean'], yerr=som_rm_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[2,1].plot(bin_centers, umap_s[metric+'_mean'], c='blue', ls='-')
    axs[2,1].errorbar(bin_centers, umap_s[metric+'_mean'], yerr=umap_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue')
    axs[2,1].plot(bin_centers, umap_lp[metric+'_mean'], c='blue', ls='--', alpha=umap_lp_alpha)
    axs[2,1].errorbar(bin_centers, umap_lp[metric+'_mean'], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c='blue', alpha=umap_lp_alpha)
    axs[2,1].axhline(y = 0.0, color = 'k', linestyle =':', lw=1.3, alpha=0.8)
    axs[2,1].set_ylim(-bias_ymax,bias_ymax)
    axs[2,1].set_xlim(bin_min,None)
    #axs[2,1].tick_params(axis='y', labelsize=18)
    #axs[2,1].tick_params(axis='x', labelsize=18)
    axs[2,1].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)
    fig.subplots_adjust(top=0.95)
    fig.legend(bbox_to_anchor=(0.03, 1.02, 1., .102), loc='center',
                        ncols=2, borderaxespad=0., fontsize=17)
    fig.tight_layout()
    #fig.suptitle('SOM vs UMAP: Statistics by Redshift', fontsize=14);
    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')

# create a Figure with performance statistics binned by redshift, corresponding to Figure 5 in our paper
def z_binned_stats3(somrm_perf, somtco_perf, umap_perf, filename=None):
    '''This function creates a multi-panel plot of the performance statistics binned by redshift, 
    corresponding to Figure 5 in our paper.

    Parameters
    ----------

    umap_errs: dictionary
        dictionary containing the redshift-binned uncertainties for UMAP, with the keys
        'outlier', 'nmad', 'bias' containing the uncertainties on the specified metric
    som_errs: dictionary
        dictionary containing the redshift-binned uncertainties for SOM, with the keys
        'outlier', 'nmad', 'bias' containing the uncertainties on the specified metric
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

    # create the figure
    fig, axs = plt.subplots(3, 2, figsize=(16,12))

    st='specz_trained'
    lpt='LePHARE_trained'

    bin_centers = umap_perf[lpt]['bin_centers'].copy()

    metric='f_outlier'
    fout_ymax = 0.86*np.max([np.max(umap_perf[st][metric]['mean']), np.max(umap_perf[lpt][metric]['mean']), 
                            np.max(somrm_perf[st][metric]['mean']), np.max(somrm_perf[lpt][metric]['mean']),
                            np.max(somtco_perf[st][metric]['mean']), 
                            np.max(somtco_perf[lpt][metric]['mean'])]) + np.max([np.max(somtco_perf['specz_trained'][metric]['unc']), 
                            np.max(somrm_perf['specz_trained'][metric]['unc']), np.max(umap_perf['specz_trained'][metric]['unc']), np.max(somtco_perf['LePHARE_trained'][metric]['unc']), 
                            np.max(somrm_perf['LePHARE_trained'][metric]['unc']), np.max(umap_perf['LePHARE_trained'][metric]['unc'])])
    metric='nmad'
    nmad_ymax = 1.01*np.max([np.max(umap_perf[st][metric]['mean']), np.max(umap_perf[lpt][metric]['mean']), 
                            np.max(somrm_perf[st][metric]['mean']), np.max(somrm_perf[lpt][metric]['mean']),
                            np.max(somtco_perf[st][metric]['mean']), 
                            np.max(somtco_perf[lpt][metric]['mean'])]) + np.max([np.max(somtco_perf['specz_trained'][metric]['unc']), 
                            np.max(somrm_perf['specz_trained'][metric]['unc']), np.max(umap_perf['specz_trained'][metric]['unc']), np.max(somtco_perf['LePHARE_trained'][metric]['unc']), 
                            np.max(somrm_perf['LePHARE_trained'][metric]['unc']), np.max(umap_perf['LePHARE_trained'][metric]['unc'])])
    metric='bias'
    bias_ymax = 1.01*np.max([np.max(umap_perf[st][metric]['mean']), np.max(umap_perf[lpt][metric]['mean']), 
                            np.max(somrm_perf[st][metric]['mean']), np.max(somrm_perf[lpt][metric]['mean']),
                            np.max(somtco_perf[st][metric]['mean']), 
                            np.max(somtco_perf[lpt][metric]['mean'])]) + np.max([np.max(somtco_perf['specz_trained'][metric]['unc']), 
                            np.max(somrm_perf['specz_trained'][metric]['unc']), np.max(umap_perf['specz_trained'][metric]['unc']), np.max(somtco_perf['LePHARE_trained'][metric]['unc']), 
                            np.max(somrm_perf['LePHARE_trained'][metric]['unc']), np.max(umap_perf['LePHARE_trained'][metric]['unc'])])

    c1, c2, c3 = 'C0', 'C1', 'C2'
    umap_lp_alpha=0.4

    #ticks
    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=18, length=6, width=1.5, 
                        bottom=True,top=False,left=True, right=False,
                        direction='out')

    # UPPER LEFT
    metric='f_outlier'
    axs[0,0].plot(bin_centers, somtco_perf[lpt][metric]['mean'], c=c3, alpha=0.7, label='SOM (LePHARE-trained, cells with training redshifts)', ls='--')
    axs[0,0].plot(bin_centers, somrm_perf[lpt][metric]['mean'], c=c2, alpha=0.7, label='SOM (LePHARE-trained, all cells with re-matching)', ls='--')
    axs[0,0].errorbar(bin_centers, somtco_perf[lpt][metric]['mean'], yerr=somtco_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[0,0].errorbar(bin_centers, somrm_perf[lpt][metric]['mean'], yerr=somrm_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[0,0].plot(bin_centers, umap_perf[lpt][metric]['mean'], c=c1, label='UMAP (LePHARE-trained)', ls='--')
    axs[0,0].errorbar(bin_centers, umap_perf[lpt][metric]['mean'], yerr=umap_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[0,0].set_ylabel(r'$f_\mathrm{outlier}$', fontsize=25)
    axs[0,0].set_ylim(0,fout_ymax)
    axs[0,0].set_xlim(0,None)
    # MIDDLE LEFT
    metric='nmad'
    axs[1,0].plot(bin_centers, somtco_perf[lpt][metric]['mean'], c=c3, alpha=0.7, ls='--')
    axs[1,0].plot(bin_centers, somrm_perf[lpt][metric]['mean'], c=c2, alpha=0.7, ls='--')
    axs[1,0].errorbar(bin_centers, somtco_perf[lpt][metric]['mean'], yerr=somtco_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[1,0].errorbar(bin_centers, somrm_perf[lpt][metric]['mean'], yerr=somrm_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[1,0].plot(bin_centers, umap_perf[lpt][metric]['mean'], c=c1, ls='--')
    axs[1,0].errorbar(bin_centers, umap_perf[lpt][metric]['mean'], yerr=umap_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[1,0].set_ylabel(r'$\sigma_\mathrm{NMAD}$', fontsize=25)
    axs[1,0].set_ylim(0,nmad_ymax)
    axs[1,0].set_xlim(0,None)
    # BOTTOM LEFT
    metric='bias'
    axs[2,0].plot(bin_centers, somtco_perf[lpt][metric]['mean'], c=c3, alpha=0.7, ls='--')
    axs[2,0].plot(bin_centers, somrm_perf[lpt][metric]['mean'], c=c2, alpha=0.7, ls='--')
    axs[2,0].errorbar(bin_centers, somtco_perf[lpt][metric]['mean'], yerr=somtco_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[2,0].errorbar(bin_centers, somrm_perf[lpt][metric]['mean'], yerr=somrm_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[2,0].plot(bin_centers, umap_perf[lpt][metric]['mean'], c=c1, ls='--')
    axs[2,0].errorbar(bin_centers, umap_perf[lpt][metric]['mean'], yerr=umap_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[2,0].axhline(y = 0.0, color = 'k', linestyle = ':', lw=1.3, alpha=0.8)
    axs[2,0].set_ylim(-bias_ymax,bias_ymax)
    axs[2,0].set_xlim(0,None)
    axs[2,0].set_ylabel('Bias ('+r'$\langle \frac{\Delta z}{1+z} \rangle$'+')', fontsize=25)
    axs[2,0].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)
    # UPPER RIGHT
    metric='f_outlier'
    axs[0,1].plot(bin_centers, umap_perf[lpt][metric]['mean'], c=c1, ls='--', alpha=umap_lp_alpha)
    axs[0,1].errorbar(bin_centers, umap_perf[lpt][metric]['mean'], yerr=umap_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1, alpha=umap_lp_alpha)
    axs[0,1].plot(bin_centers, somtco_perf[st][metric]['mean'], c=c3, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, cells with training redshifts)', ls='-')
    axs[0,1].plot(bin_centers, somrm_perf[st][metric]['mean'], c=c2, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, all cells with re-matching)', ls='-')
    axs[0,1].errorbar(bin_centers, somtco_perf[st][metric]['mean'], yerr=somtco_perf['specz_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[0,1].errorbar(bin_centers, somrm_perf[st][metric]['mean'], yerr=somrm_perf['specz_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[0,1].plot(bin_centers, umap_perf[st][metric]['mean'], c=c1, label='UMAP (spec-'+r'$z$'+'-trained)', ls='-')
    axs[0,1].errorbar(bin_centers, umap_perf[st][metric]['mean'], yerr=umap_perf['specz_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[0,1].set_ylim(0,fout_ymax)
    axs[0,1].set_xlim(0,None)
    # MIDDLE RIGHT
    metric='nmad'
    axs[1,1].plot(bin_centers, somtco_perf[st][metric]['mean'], c=c3, alpha=0.7, ls='-')
    axs[1,1].plot(bin_centers, somrm_perf[st][metric]['mean'], c=c2, alpha=0.7, ls='-')
    axs[1,1].errorbar(bin_centers, somtco_perf[st][metric]['mean'], yerr=somtco_perf['specz_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[1,1].errorbar(bin_centers, somrm_perf[st][metric]['mean'], yerr=somrm_perf['specz_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[1,1].plot(bin_centers, umap_perf[st][metric]['mean'], c=c1, ls='-')
    axs[1,1].errorbar(bin_centers, umap_perf[st][metric]['mean'], yerr=umap_perf['specz_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[1,1].plot(bin_centers, umap_perf[lpt][metric]['mean'], c=c1, ls='--', alpha=umap_lp_alpha)
    axs[1,1].errorbar(bin_centers, umap_perf[lpt][metric]['mean'], yerr=umap_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1, alpha=umap_lp_alpha)
    axs[1,1].set_ylim(0,nmad_ymax)
    axs[1,1].set_xlim(0,None)
    # BOTTOM RIGHT
    metric='bias'
    axs[2,1].plot(bin_centers, somtco_perf[st][metric]['mean'], c=c3, alpha=0.7, ls='-')
    axs[2,1].plot(bin_centers, somrm_perf[st][metric]['mean'], c=c2, alpha=0.7, ls='-')
    axs[2,1].errorbar(bin_centers, somtco_perf[st][metric]['mean'], yerr=somtco_perf['specz_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c3)
    axs[2,1].errorbar(bin_centers, somrm_perf[st][metric]['mean'], yerr=somrm_perf['specz_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[2,1].plot(bin_centers, umap_perf[st][metric]['mean'], c=c1, ls='-')
    axs[2,1].errorbar(bin_centers, umap_perf[st][metric]['mean'], yerr=umap_perf['specz_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[2,1].plot(bin_centers, umap_perf[lpt][metric]['mean'], c=c1, ls='--', alpha=umap_lp_alpha)
    axs[2,1].errorbar(bin_centers, umap_perf[lpt][metric]['mean'], yerr=umap_perf['LePHARE_trained'][metric]['unc'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1, alpha=umap_lp_alpha)
    axs[2,1].axhline(y = 0.0, color = 'k', linestyle =':', lw=1.3, alpha=0.8)
    axs[2,1].set_ylim(-bias_ymax,bias_ymax)
    axs[2,1].set_xlim(0,None)
    axs[2,1].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)
    fig.subplots_adjust(top=0.95)
    fig.legend(bbox_to_anchor=(0.03, 1.02, 1., .102), loc='center',
                        ncols=2, borderaxespad=0., fontsize=17)
    fig.tight_layout()

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
    axs[0].plot(bin_centers, outliers_s, c='0.5', label='Color Space (spec-'+r'$z$'+'-trained)')
    axs[0].plot(bin_centers, lp_outliers, c='0.5', label='Color Space (LePHARE-trained)', ls='--')
    axs[0].plot(bin_centers, umap_outliers_s, c='blue', label='UMAP (spec-'+r'$z$'+'-trained)')
    axs[0].plot(bin_centers, umap_outliers, c='blue', label='UMAP (LePHARE-trained)', ls='--')
    axs[0].scatter(bin_centers, outliers_s, c='0.5', s=20)
    axs[0].scatter(bin_centers, lp_outliers, c='0.5', s=20)
    axs[0].scatter(bin_centers, umap_outliers_s, c='blue', s=20)
    axs[0].scatter(bin_centers, umap_outliers, c='blue', s=20)
    axs[0].set_ylabel(r'$f_\mathrm{outlier}$', fontsize=25)
    axs[0].tick_params(axis='y', labelsize=18)
    axs[0].tick_params(axis='x', labelsize=18)
    axs[0].set_ylim(0,fout_ymax)
    axs[0].set_xlim(0,None)
    axs[0].legend(fontsize=18)

    # MIDDLE LEFT
    axs[1].plot(bin_centers, nmads_s, c='0.5', label='Color Space (spec-'+r'$z$'+'-trained)')
    axs[1].plot(bin_centers, lp_nmads, c='0.5', label='Color Space (LePHARE-trained)', ls='--')
    axs[1].plot(bin_centers, umap_nmads_s, c='blue', label='UMAP (spec-'+r'$z$'+'-trained)')
    axs[1].plot(bin_centers, umap_nmads, c='blue', label='UMAP (LePHARE-trained)', ls='--')
    axs[1].scatter(bin_centers, nmads_s, c='0.5', s=20)
    axs[1].scatter(bin_centers, lp_nmads, c='0.5', s=20)
    axs[1].scatter(bin_centers, umap_nmads_s, c='blue', s=20)
    axs[1].scatter(bin_centers, umap_nmads, c='blue', s=20)
    axs[1].set_ylabel(r'$\sigma_\mathrm{NMAD}$', fontsize=25)
    axs[1].tick_params(axis='y', labelsize=18)
    axs[1].tick_params(axis='x', labelsize=18)
    axs[1].set_ylim(0,nmad_ymax)
    axs[1].set_xlim(0,None)

    # BOTTOM LEFT
    axs[2].plot(bin_centers, biases_s, c='0.5', label='Color Space (CSRC-trained)')
    axs[2].plot(bin_centers, lp_biases, c='0.5', label='Color Space (LePHARE-trained)', ls='--')
    axs[2].plot(bin_centers, umap_biases_s, c='blue', label='UMAP (CSRC-trained)')
    axs[2].plot(bin_centers, umap_biases, c='blue', label='UMAP (LePHARE-trained)', ls='--')
    axs[2].scatter(bin_centers, biases_s, c='0.5', s=20)
    axs[2].scatter(bin_centers, lp_biases, c='0.5', s=20)
    axs[2].scatter(bin_centers, umap_biases_s, c='blue', s=20)
    axs[2].scatter(bin_centers, umap_biases, c='blue', s=20)
    axs[2].axhline(y = 0.0, color = 'k', linestyle = '-.', lw=1.3, alpha=0.6)
    axs[2].set_ylim(-bias_ymax,bias_ymax)
    axs[2].set_xlim(0,None)
    axs[2].set_ylabel('Bias ('+r'$\langle \frac{\Delta z}{1+z} \rangle$'+')', fontsize=25)
    axs[2].tick_params(axis='y', labelsize=18)
    axs[2].tick_params(axis='x', labelsize=18)
    axs[2].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)

    fig.tight_layout()
    #fig.suptitle('SOM vs UMAP: Statistics by Redshift', fontsize=14);
    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')

# create a plot similar to Figure 5, comparing the UMAP and input color space performance
def z_binned_umap_vs_colors2(umap_perf, colors_perf, umap_errs, colors_errs, bin_min=0.0, filename=None):
    '''This function creates a multi-panel plot of the performance statistics binned by redshift, 
    corresponding to Figure 5 in our paper.

    Parameters
    ----------
    fullcat_lp: pandas DataFrame
        DataFrame containing the full catalog of test objects, LePHARE-trained case
    umap_errs: dictionary
        dictionary containing the redshift-binned uncertainties for UMAP, with the keys
        'outlier', 'nmad', 'bias' containing the uncertainties on the specified metric
    bin_min: float, default=0.0
        the lower limit of the redshift binning, and the lower x-axis limit for plotting
    filename: string, default=None
        optional filename, if !=None the plot will be saved according to the format in the 
        string, e.g. 'z_binned_figure.png'

    Returns
    ----------
    figure: the plot!
    optionally saved to directory if filename!=None
    '''

    # create the figure
    fig, axs = plt.subplots(3, 1, figsize=(11,14))

    bin_centers = umap_perf['LePHARE_trained']['bin_centers']
    lsy = 28
    lsx = 28
    fs = 35

    metric='outliers'
    metric2='f_outlier'
    fout_ymax = 1.01*np.max([np.max(umap_perf['specz_trained'][metric2]['mean']), np.max(umap_perf['LePHARE_trained'][metric2]['mean']), 
                            np.max(colors_perf['specz_trained'][metric2][metric2]), 
                            np.max(colors_perf['LePHARE_trained'][metric2][metric2])]) + np.max([np.max(colors_errs[metric]), 
                            np.max(umap_errs[metric]), np.max(colors_errs[metric+'_lp']), 
                            np.max(umap_errs[metric+'_lp'])])
    metric='nmads'
    metric2='nmad'
    nmad_ymax = 1.01*np.max([np.max(umap_perf['specz_trained'][metric2]['mean']), np.max(umap_perf['LePHARE_trained'][metric2]['mean']), 
                            np.max(colors_perf['specz_trained'][metric2][metric2]), 
                            np.max(colors_perf['LePHARE_trained'][metric2][metric2])]) + np.max([np.max(colors_errs[metric]), 
                            np.max(umap_errs[metric]), np.max(colors_errs[metric+'_lp']), 
                            np.max(umap_errs[metric+'_lp'])])
    metric='biases'
    metric2='bias'
    bias_ymax = 1.01*np.max([np.max(umap_perf['specz_trained'][metric2]['mean']), np.max(umap_perf['LePHARE_trained'][metric2]['mean']), 
                            np.max(colors_perf['specz_trained'][metric2][metric2]), 
                            np.max(colors_perf['LePHARE_trained'][metric2][metric2])]) + np.max([np.max(colors_errs[metric]), 
                            np.max(umap_errs[metric]), np.max(colors_errs[metric+'_lp']), 
                            np.max(umap_errs[metric+'_lp'])])

    c1, c2 = 'C0', 'C3'

    #ticks
    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=18, length=7, width=1.5, 
                        bottom=True,top=False,left=True, right=False,
                        direction='out')
        
    axs[0].yaxis.set_major_locator(MultipleLocator(0.04))
    #axs[1].yaxis.set_major_locator(MultipleLocator(0.01))
    axs[2].yaxis.set_major_locator(MultipleLocator(0.05))

    # UPPER LEFT
    metric='outliers'
    metric2='f_outlier'
    axs[0].plot(bin_centers, colors_perf['LePHARE_trained'][metric2][metric2], c=c2, alpha=0.7, label='colors-'+r'$k$'+'NN-'+r'$z$'+' (LePHARE-trained)', ls='--')
    axs[0].plot(bin_centers, colors_perf['specz_trained'][metric2][metric2], c=c2, alpha=0.7, label='colors-'+r'$k$'+'NN-'+r'$z$'+' (spec-'+r'$z$'+'-trained)', ls='-')
    axs[0].errorbar(bin_centers, colors_perf['LePHARE_trained'][metric2][metric2], yerr=colors_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[0].errorbar(bin_centers, colors_perf['specz_trained'][metric2][metric2], yerr=colors_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[0].plot(bin_centers, umap_perf['LePHARE_trained'][metric2]['mean'], c=c1, alpha=0.7, label='UMAP (LePHARE-trained)', ls='--')
    axs[0].errorbar(bin_centers, umap_perf['LePHARE_trained'][metric2]['mean'], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[0].plot(bin_centers, umap_perf['specz_trained'][metric2]['mean'], c=c1, alpha=0.7, label='UMAP (spec-'+r'$z$'+'-trained)', ls='-')
    axs[0].errorbar(bin_centers, umap_perf['specz_trained'][metric2]['mean'], yerr=umap_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[0].set_ylabel(r'$f_\mathrm{outlier}$', fontsize=fs)
    axs[0].tick_params(axis='y', labelsize=lsy)
    axs[0].tick_params(axis='x', labelsize=lsx)
    axs[0].set_ylim(0,fout_ymax)
    axs[0].set_xlim(None,None)
    # MIDDLE LEFT
    metric='nmads'
    metric2='nmad'
    axs[1].plot(bin_centers, colors_perf['LePHARE_trained'][metric2][metric2], c=c2, alpha=0.7, ls='--')
    axs[1].plot(bin_centers, colors_perf['specz_trained'][metric2][metric2], c=c2, alpha=0.7, ls='-')
    axs[1].errorbar(bin_centers, colors_perf['LePHARE_trained'][metric2][metric2], yerr=colors_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[1].errorbar(bin_centers, colors_perf['specz_trained'][metric2][metric2], yerr=colors_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[1].plot(bin_centers, umap_perf['LePHARE_trained'][metric2]['mean'], c=c1, alpha=0.7, ls='--')
    axs[1].errorbar(bin_centers, umap_perf['LePHARE_trained'][metric2]['mean'], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[1].plot(bin_centers, umap_perf['specz_trained'][metric2]['mean'], c=c1, alpha=0.7, ls='-')
    axs[1].errorbar(bin_centers, umap_perf['specz_trained'][metric2]['mean'], yerr=umap_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[1].set_ylabel(r'$\sigma_\mathrm{NMAD}$', fontsize=fs)
    axs[1].tick_params(axis='y', labelsize=lsy)
    axs[1].tick_params(axis='x', labelsize=lsx)
    axs[1].set_ylim(0,nmad_ymax)
    axs[1].set_xlim(None,None)
    # BOTTOM LEFT
    metric='biases'
    metric2='bias'
    axs[2].plot(bin_centers, colors_perf['LePHARE_trained'][metric2][metric2], c=c2, alpha=0.7, ls='--')
    axs[2].plot(bin_centers, colors_perf['specz_trained'][metric2][metric2], c=c2, alpha=0.7, ls='-')
    axs[2].errorbar(bin_centers, colors_perf['LePHARE_trained'][metric2][metric2], yerr=colors_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[2].errorbar(bin_centers, colors_perf['specz_trained'][metric2][metric2], yerr=colors_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c2)
    axs[2].plot(bin_centers, umap_perf['LePHARE_trained'][metric2]['mean'], c=c1, alpha=0.7, ls='--')
    axs[2].errorbar(bin_centers, umap_perf['LePHARE_trained'][metric2]['mean'], yerr=umap_errs[metric+'_lp'], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[2].plot(bin_centers, umap_perf['specz_trained'][metric2]['mean'], c=c1, alpha=0.7, ls='-')
    axs[2].errorbar(bin_centers, umap_perf['specz_trained'][metric2]['mean'], yerr=umap_errs[metric], fmt='o', capsize=3,
                    lw=1, ms=4, c=c1)
    axs[2].set_ylabel('Bias ('+r'$\langle \frac{\Delta z}{1+z} \rangle$'+')', fontsize=fs)
    axs[2].tick_params(axis='y', labelsize=lsy)
    axs[2].tick_params(axis='x', labelsize=lsx)
    axs[2].set_ylim(-bias_ymax,bias_ymax)
    axs[2].axhline(y = 0.0, color = 'k', linestyle =':', lw=1.3, alpha=0.8)
    axs[2].set_xlim(None,None)
    axs[2].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=fs)
    fig.subplots_adjust(top=0.95)
    fig.legend(bbox_to_anchor=(0.03, 1.03, 1., .102), loc='center',
                        ncols=1, borderaxespad=0.0, fontsize=28)
    fig.tight_layout()
    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')

def sandplot_som(uncs, unc_sources, labels, cs, ylims=None, filename=None):
    '''
    explanation
    '''
    fig, axs = plt.subplots(3, 2, figsize=(16,12))
    trains = ['LePHARE', 'specz']
    c1, c2, c3, c4, c5 = cs
    l1, l2, l3, l4, l5 = labels
    bin_centers = uncs[trains[0]+'_trained']['bin_centers']
    metrics = ['f_outlier', 'nmad', 'bias']
    ylabels = [
        'Contribution to '+r'$f_\mathrm{outlier}$'+' Uncertainty',
        'Contribution to '+r'$\sigma_\mathrm{NMAD}$'+' Uncertainty',
        'Contribution to Bias Uncertainty'
    ]
    alpha=0.5

    #ticks
    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=18, length=6, width=1.5, 
                        bottom=True,top=False,left=True, right=False,
                        direction='out')
        #ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    #axs[0,0].yaxis.set_major_locator(MultipleLocator(0.02))
    #axs[0,1].yaxis.set_major_locator(MultipleLocator(0.01))
    #axs[0,2].yaxis.set_major_locator(MultipleLocator(0.02))
    #axs[1,0].yaxis.set_major_locator(MultipleLocator(0.02))
    #axs[1,1].yaxis.set_major_locator(MultipleLocator(0.01))
    #axs[1,2].yaxis.set_major_locator(MultipleLocator(0.02))

    # OUTLIERS
    for i, train in enumerate(trains):
        for j, metric in enumerate(metrics):
            if i==0 and j==0:
                l1, l2, l3, l4, l5 = labels
            else:
                l1, l2, l3, l4, l5 = None, None, None, None, None
            if i==0:
                axs[j,i].set_ylabel(ylabels[j], fontsize=16)
            unc1, unc2, unc3, unc4, unc5 = [uncs[train+'_trained'][metric]['unc_'+unc_source] for unc_source in unc_sources]

            axs[j,i].plot(bin_centers, unc1, c=c1, ls='-', label=l1)
            axs[j,i].fill_between(bin_centers, unc1, alpha=alpha, color=c1)
            axs[j,i].plot(bin_centers, unc2+unc1, c=c2, ls='-', label=l2)
            axs[j,i].fill_between(bin_centers, unc2+unc1, y2=unc1, alpha=alpha, color=c2)
            axs[j,i].plot(bin_centers, unc3+unc2+unc1, c=c3, ls='-', label=l3)
            axs[j,i].fill_between(bin_centers, unc3+unc2+unc1,y2=unc2+unc1, alpha=alpha, color=c3)
            axs[j,i].plot(bin_centers, unc4+unc3+unc2+unc1, c=c4, ls='-', label=l4)
            axs[j,i].fill_between(bin_centers, unc4+unc3+unc2+unc1,y2=unc3+unc2+unc1, alpha=alpha, color=c4)
            axs[j,i].plot(bin_centers, unc5+unc4+unc3+unc2+unc1, c=c5, ls='-', label=l5)
            axs[j,i].fill_between(bin_centers, unc5+unc4+unc3+unc2+unc1,y2=unc4+unc3+unc2+unc1, alpha=alpha, color=c5)
            axs[j,i].set_ylim(0, ylims[j])
        axs[2,i].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)

    axs[0,0].set_title('LePHARE-trained', fontsize=25)
    axs[0,1].set_title('spec-'+r'$z$'+'-trained', fontsize=25)
    fig.subplots_adjust(top=0.95)
    fig.legend(bbox_to_anchor=(0.03, 0.98, 1., .102), loc='center',
                        ncols=3, borderaxespad=0., fontsize=22)
    fig.tight_layout()
    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')

def sandplot_umap(uncs, unc_sources, labels, cs, ylims=None, filename=None):
    '''
    explanation
    '''
    fig, axs = plt.subplots(3, 2, figsize=(16,12))
    trains = ['LePHARE', 'specz']
    c1, c2, c3, c4 = cs
    l1, l2, l3, l4 = labels
    bin_centers = uncs[trains[0]+'_trained']['bin_centers']
    metrics = ['f_outlier', 'nmad', 'bias']
    ylabels = [
        'Contribution to '+r'$f_\mathrm{outlier}$'+' Uncertainty',
        'Contribution to '+r'$\sigma_\mathrm{NMAD}$'+' Uncertainty',
        'Contribution to Bias Uncertainty'
    ]
    alpha=0.5

    #ticks
    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=18, length=6, width=1.5, 
                        bottom=True,top=False,left=True, right=False,
                        direction='out')

    # OUTLIERS
    for i, train in enumerate(trains):
        for j, metric in enumerate(metrics):
            if i==0 and j==0:
                l1, l2, l3, l4 = labels
            else:
                l1, l2, l3, l4 = None, None, None, None
            if i==0:
                axs[j,i].set_ylabel(ylabels[j], fontsize=16)
            unc1, unc2, unc3, unc4 = [uncs[train+'_trained'][metric]['unc_'+unc_source] for unc_source in unc_sources]

            axs[j,i].plot(bin_centers, unc1, c=c1, ls='-', label=l1)
            axs[j,i].fill_between(bin_centers, unc1, alpha=alpha, color=c1)
            axs[j,i].plot(bin_centers, unc2+unc1, c=c2, ls='-', label=l2)
            axs[j,i].fill_between(bin_centers, unc2+unc1, y2=unc1, alpha=alpha, color=c2)
            axs[j,i].plot(bin_centers, unc3+unc2+unc1, c=c3, ls='-', label=l3)
            axs[j,i].fill_between(bin_centers, unc3+unc2+unc1,y2=unc2+unc1, alpha=alpha, color=c3)
            axs[j,i].plot(bin_centers, unc4+unc3+unc2+unc1, c=c4, ls='-', label=l4)
            axs[j,i].fill_between(bin_centers, unc4+unc3+unc2+unc1,y2=unc3+unc2+unc1, alpha=alpha, color=c4)
            axs[j,i].set_ylim(0, ylims[j])
        axs[2,i].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)

    axs[0,0].set_title('LePHARE-trained', fontsize=25)
    axs[0,1].set_title('spec-'+r'$z$'+'-trained', fontsize=25)
    fig.subplots_adjust(top=0.95)
    fig.legend(bbox_to_anchor=(0.03, 0.98, 1., .102), loc='center',
                        ncols=3, borderaxespad=0., fontsize=22)
    fig.tight_layout()
    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')

def sandplot_colors(uncs, unc_sources, labels, cs, ylims=None, filename=None):
    '''
    explanation
    '''
    fig, axs = plt.subplots(3, 2, figsize=(16,12))
    trains = ['LePHARE', 'specz']
    c1, c2 = cs
    l1, l2 = labels
    bin_centers = uncs[trains[0]+'_trained']['bin_centers']
    metrics = ['f_outlier', 'nmad', 'bias']
    ylabels = [
        'Contribution to '+r'$f_\mathrm{outlier}$'+' Uncertainty',
        'Contribution to '+r'$\sigma_\mathrm{NMAD}$'+' Uncertainty',
        'Contribution to Bias Uncertainty'
    ]
    alpha=0.5

    #ticks
    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=18, length=6, width=1.5, 
                        bottom=True,top=False,left=True, right=False,
                        direction='out')
        #ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    #axs[0,0].yaxis.set_major_locator(MultipleLocator(0.02))
    #axs[0,1].yaxis.set_major_locator(MultipleLocator(0.01))
    #axs[0,2].yaxis.set_major_locator(MultipleLocator(0.02))
    #axs[1,0].yaxis.set_major_locator(MultipleLocator(0.02))
    #axs[1,1].yaxis.set_major_locator(MultipleLocator(0.01))
    #axs[1,2].yaxis.set_major_locator(MultipleLocator(0.02))

    # OUTLIERS
    for i, train in enumerate(trains):
        for j, metric in enumerate(metrics):
            if i==0 and j==0:
                l1, l2 = labels
            else:
                l1, l2 = None, None
            if i==0:
                axs[j,i].set_ylabel(ylabels[j], fontsize=16)
            unc1, unc2 = [uncs[train+'_trained'][metric]['unc_'+unc_source] for unc_source in unc_sources]

            axs[j,i].plot(bin_centers, unc1, c=c1, ls='-', label=l1)
            axs[j,i].fill_between(bin_centers, unc1, alpha=alpha, color=c1)
            axs[j,i].plot(bin_centers, unc2+unc1, c=c2, ls='-', label=l2)
            axs[j,i].fill_between(bin_centers, unc2+unc1, y2=unc1, alpha=alpha, color=c2)
            axs[j,i].set_ylim(0, ylims[j])
        axs[2,i].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)

    axs[0,0].set_title('LePHARE-trained', fontsize=25)
    axs[0,1].set_title('spec-'+r'$z$'+'-trained', fontsize=25)
    fig.subplots_adjust(top=0.95)
    fig.legend(bbox_to_anchor=(0.03, 0.98, 1., .102), loc='center',
                        ncols=3, borderaxespad=0., fontsize=22)
    fig.tight_layout()
    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')

def sandplot_split(uncs, unc_sources, labels, cs, ylims=None, filename=None):

    trains = ['LePHARE', 'specz']
    metrics = ['f_outlier', 'nmad', 'bias']

    ylabels = [
        r'$f_\mathrm{outlier}$ Uncertainty',
        r'$\sigma_\mathrm{NMAD}$ Uncertainty',
        'Bias Uncertainty'
    ]

    alpha = 0.5
    n_uncs = len(unc_sources)

    fig = plt.figure(figsize=(17, 14))

    outer = fig.add_gridspec(nrows=3,ncols=2,wspace=0.07,hspace=0.18)

    # Store panels as a 3x2 array
    all_axes = np.empty((3, 2), dtype=object)

    # Create nested axes
    for row in range(3):
        for col in range(2):

            inner = outer[row, col].subgridspec(nrows=n_uncs,ncols=1,hspace=0.0)

            panel_axes = [fig.add_subplot(inner[0])]
            panel_axes += [
                fig.add_subplot(inner[k], sharex=panel_axes[0])
                for k in range(1, n_uncs)
            ]

            # Hide x tick labels on upper axes within each panel
            for ax in panel_axes[:-1]:
                ax.tick_params(labelbottom=False)

            # Tick styling
            for ax in panel_axes:
                ax.tick_params(axis='x',labelsize=17,length=5,width=1.2,
                    bottom=True,top=False,left=True,right=False,direction='out'
                )
                ax.tick_params(axis='y',labelsize=16,length=5,width=1.2,
                    bottom=True,top=False,left=True,right=False,direction='out'
                )

            if col == 0:
                boxes = [ax.get_position() for ax in panel_axes]
                bbox = Bbox.union(boxes)
                x = bbox.x0 - 0.03
                y = (bbox.y0 + bbox.y1) / 2
                fig.text(x, y, ylabels[row], rotation=90,
                    va='center', ha='right', fontsize=21)

            if col == 1:
                for ax in panel_axes:
                    ax.tick_params(axis='y', left=True, labelleft=False)

            # Only bottom row gets x labels
            if row == 2:
                panel_axes[-1].set_xlabel(
                    r'$z_\mathrm{LePHARE}$',
                    fontsize=22
                )

            all_axes[row, col] = panel_axes

    # Plot data
    bin_centers = uncs['LePHARE_trained']['bin_centers']

    for i, train in enumerate(trains):
        for j, metric in enumerate(metrics):

            panel_axes = all_axes[j, i]

            unc_arrays = [
                uncs[f'{train}_trained'][metric][f'unc_{src}']
                for src in unc_sources
            ]

            for k, (ax, unc, label, color) in enumerate(
                zip(panel_axes, unc_arrays, labels, cs)
            ):

                # Only label once for legend
                legend_label = label if (i == 0 and j == 0) else None

                ax.plot(bin_centers,unc,color=color,lw=2,label=legend_label)

                ax.fill_between(bin_centers,unc,color=color,alpha=alpha)

                if ylims is not None:
                    if np.ndim(ylims) == 1:
                        ax.set_ylim(0, ylims[j])
                    else:
                        ax.set_ylim(0, ylims[j][k])

    custom_ticks = {
    (0, 0, -1): [0.00, 0.02],
    (1, 0, -1): [0.00, 0.01],  # middle-left bottom axis
    (2, 0, -1): [0.00, 0.02],  # bottom-left bottom axis
    }
    for (r, c, k), ticks in custom_ticks.items():
        for ax in all_axes[r, c]:
            ax.set_yticks(ticks)
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ------------------------------------------------------------------
    # Titles
    all_axes[0, 0][0].set_title('LePHARE-trained',fontsize=21,pad=10)
    all_axes[0, 1][0].set_title(r'spec-$z$-trained',fontsize=21,pad=10)

    # Legend
    handles = []
    leg_labels = []

    for ax in all_axes[0, 0]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        leg_labels.extend(l)

    fig.legend(handles,leg_labels,bbox_to_anchor=(0.05, 0.95, 0.9, 0.05),
        loc='upper center',ncols=3,fontsize=21,frameon=True)
    
    fig.subplots_adjust(left=0.13)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

    #return fig, all_axes

def total_uncertainties(somrm_perf, somtco_perf, umap_perf, colors_perf, filename=None):
    '''This function creates a multi-panel plot of the performance statistics binned by redshift, 
    corresponding to Figure 5 in our paper.

    Parameters
    ----------
    umap_errs: dictionary
        dictionary containing the redshift-binned uncertainties for UMAP, with the keys
        'outlier', 'nmad', 'bias' containing the uncertainties on the specified metric
    som_errs: dictionary
        dictionary containing the redshift-binned uncertainties for SOM, with the keys
        'outlier', 'nmad', 'bias' containing the uncertainties on the specified metric
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

    # create the figure
    fig, axs = plt.subplots(3, 2, figsize=(16,12))

    #axs[0,0].yaxis.set_major_locator(MultipleLocator(0.02))
    #axs[0,1].yaxis.set_major_locator(MultipleLocator(0.002))
    #axs[0,2].yaxis.set_major_locator(MultipleLocator(0.02))
    axs[1,0].yaxis.set_major_locator(MultipleLocator(0.004))
    axs[1,1].yaxis.set_major_locator(MultipleLocator(0.004))
    #axs[1,2].yaxis.set_major_locator(MultipleLocator(0.02))

    st='specz_trained'
    lpt='LePHARE_trained'

    bin_centers = umap_perf[lpt]['bin_centers'].copy()

    metric='f_outlier'
    fout_ymax = 1.05*np.max([np.max(somtco_perf['specz_trained'][metric]['unc']), 
                            np.max(somrm_perf['specz_trained'][metric]['unc']), np.max(umap_perf['specz_trained'][metric]['unc']), np.max(somtco_perf['LePHARE_trained'][metric]['unc']), 
                            np.max(somrm_perf['LePHARE_trained'][metric]['unc']), np.max(umap_perf['LePHARE_trained'][metric]['unc'])])
    metric='nmad'
    nmad_ymax = 1.05*np.max([np.max(somtco_perf['specz_trained'][metric]['unc']), 
                            np.max(somrm_perf['specz_trained'][metric]['unc']), np.max(umap_perf['specz_trained'][metric]['unc']), np.max(somtco_perf['LePHARE_trained'][metric]['unc']), 
                            np.max(somrm_perf['LePHARE_trained'][metric]['unc']), np.max(umap_perf['LePHARE_trained'][metric]['unc'])])
    metric='bias'
    bias_ymax = 1.05*np.max([np.max(somtco_perf['specz_trained'][metric]['unc']), 
                            np.max(somrm_perf['specz_trained'][metric]['unc']), np.max(umap_perf['specz_trained'][metric]['unc']), np.max(somtco_perf['LePHARE_trained'][metric]['unc']), 
                            np.max(somrm_perf['LePHARE_trained'][metric]['unc']), np.max(umap_perf['LePHARE_trained'][metric]['unc'])])

    c1, c2, c3, c4 = 'C0', 'C1', 'C2', 'C3'
    #umap_lp_alpha=0.4
    dotsize = 20

    #ticks
    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=18, length=6, width=1.5, 
                        bottom=True,top=False,left=True, right=False,
                        direction='out')

    # UPPER LEFT
    metric='f_outlier'
    axs[0,0].plot(bin_centers, somtco_perf[lpt][metric]['unc'], c=c3, alpha=0.7, label='SOM (LePHARE-trained, cells with training redshifts)', ls='--')
    axs[0,0].plot(bin_centers, somrm_perf[lpt][metric]['unc'], c=c2, alpha=0.7, label='SOM (LePHARE-trained, all cells with re-matching)', ls='--')
    axs[0,0].scatter(bin_centers, somtco_perf[lpt][metric]['unc'], c=c3, s=dotsize)
    axs[0,0].scatter(bin_centers, somrm_perf[lpt][metric]['unc'], c=c2, s=dotsize)
    axs[0,0].plot(bin_centers, umap_perf[lpt][metric]['unc'], c=c1, label='UMAP (LePHARE-trained)', ls='--')
    axs[0,0].scatter(bin_centers, umap_perf[lpt][metric]['unc'], c=c1, s=dotsize)
    axs[0,0].plot(bin_centers, colors_perf['outliers_lp'], c=c4, label='colors-'+r'$k$'+'NN-'+r'$z$'+' (LePHARE-trained)', ls='--')
    axs[0,0].scatter(bin_centers, colors_perf['outliers_lp'], c=c4, s=dotsize)
    axs[0,0].set_ylabel(r'$f_\mathrm{outlier}$ Uncertainty', fontsize=25)
    axs[0,0].set_ylim(0,fout_ymax)
    #axs[0,0].set_xlim(0,None)
    # MIDDLE LEFT
    metric='nmad'
    axs[1,0].plot(bin_centers, somtco_perf[lpt][metric]['unc'], c=c3, alpha=0.7, ls='--')
    axs[1,0].plot(bin_centers, somrm_perf[lpt][metric]['unc'], c=c2, alpha=0.7, ls='--')
    axs[1,0].scatter(bin_centers, somtco_perf[lpt][metric]['unc'], c=c3, s=dotsize)
    axs[1,0].scatter(bin_centers, somrm_perf[lpt][metric]['unc'], c=c2, s=dotsize)
    axs[1,0].plot(bin_centers, umap_perf[lpt][metric]['unc'], c=c1, ls='--')
    axs[1,0].scatter(bin_centers, umap_perf[lpt][metric]['unc'], c=c1, s=dotsize)
    axs[1,0].plot(bin_centers, colors_perf['nmads_lp'], c=c4, ls='--')
    axs[1,0].scatter(bin_centers, colors_perf['nmads_lp'], c=c4, s=dotsize)
    axs[1,0].set_ylabel(r'$\sigma_\mathrm{NMAD}$ Uncertainty', fontsize=25)
    axs[1,0].set_ylim(0,nmad_ymax)
    #axs[1,0].set_xlim(0,None)
    # BOTTOM LEFT
    metric='bias'
    axs[2,0].plot(bin_centers, somtco_perf[lpt][metric]['unc'], c=c3, alpha=0.7, ls='--')
    axs[2,0].plot(bin_centers, somrm_perf[lpt][metric]['unc'], c=c2, alpha=0.7, ls='--')
    axs[2,0].scatter(bin_centers, somtco_perf[lpt][metric]['unc'], c=c3, s=dotsize)
    axs[2,0].scatter(bin_centers, somrm_perf[lpt][metric]['unc'], c=c2, s=dotsize)
    axs[2,0].plot(bin_centers, umap_perf[lpt][metric]['unc'], c=c1, ls='--')
    axs[2,0].scatter(bin_centers, umap_perf[lpt][metric]['unc'], c=c1, s=dotsize)
    axs[2,0].plot(bin_centers, colors_perf['biases_lp'], c=c4, ls='--')
    axs[2,0].scatter(bin_centers, colors_perf['biases_lp'], c=c4, s=dotsize)
    axs[2,0].set_ylim(0,bias_ymax)
    #axs[2,0].set_xlim(0,None)
    axs[2,0].set_ylabel('Bias Uncertainty', fontsize=25)
    axs[2,0].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)
    # UPPER RIGHT
    metric='f_outlier'
    axs[0,1].plot(bin_centers, somtco_perf[st][metric]['unc'], c=c3, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, cells with training redshifts)', ls='-')
    axs[0,1].plot(bin_centers, somrm_perf[st][metric]['unc'], c=c2, alpha=0.7, label='SOM (spec-'+r'$z$'+'-trained, all cells with re-matching)', ls='-')
    axs[0,1].scatter(bin_centers, somtco_perf[st][metric]['unc'], c=c3, s=dotsize)
    axs[0,1].scatter(bin_centers, somrm_perf[st][metric]['unc'], c=c2, s=dotsize)
    axs[0,1].plot(bin_centers, umap_perf[st][metric]['unc'], c=c1, label='UMAP (spec-'+r'$z$'+'-trained)', ls='-')
    axs[0,1].scatter(bin_centers, umap_perf[st][metric]['unc'], c=c1, s=dotsize)
    axs[0,1].plot(bin_centers, colors_perf['outliers'], c=c4, label='colors-'+r'$k$'+'NN-'+r'$z$'+' (spec-'+r'$z$'+'-trained)', ls='-')
    axs[0,1].scatter(bin_centers, colors_perf['outliers'], c=c4, s=dotsize)
    axs[0,1].set_ylim(0,fout_ymax)
    #axs[0,1].set_xlim(0,None)
    # MIDDLE RIGHT
    metric='nmad'
    axs[1,1].plot(bin_centers, somtco_perf[st][metric]['unc'], c=c3, alpha=0.7, ls='-')
    axs[1,1].plot(bin_centers, somrm_perf[st][metric]['unc'], c=c2, alpha=0.7, ls='-')
    axs[1,1].scatter(bin_centers, somtco_perf[st][metric]['unc'], c=c3, s=dotsize)
    axs[1,1].scatter(bin_centers, somrm_perf[st][metric]['unc'], c=c2, s=dotsize)
    axs[1,1].plot(bin_centers, umap_perf[st][metric]['unc'], c=c1, ls='-')
    axs[1,1].scatter(bin_centers, umap_perf[st][metric]['unc'], c=c1, s=dotsize)
    axs[1,1].plot(bin_centers, colors_perf['nmads'], c=c4, ls='-')
    axs[1,1].scatter(bin_centers, colors_perf['nmads'], c=c4, s=dotsize)
    axs[1,1].set_ylim(0,nmad_ymax)
    #axs[1,1].set_xlim(0,None)
    # BOTTOM RIGHT
    metric='bias'
    axs[2,1].plot(bin_centers, somtco_perf[st][metric]['unc'], c=c3, alpha=0.7, ls='-')
    axs[2,1].plot(bin_centers, somrm_perf[st][metric]['unc'], c=c2, alpha=0.7, ls='-')
    axs[2,1].scatter(bin_centers, somtco_perf[st][metric]['unc'], c=c3, s=dotsize)
    axs[2,1].scatter(bin_centers, somrm_perf[st][metric]['unc'], c=c2, s=dotsize)
    axs[2,1].plot(bin_centers, umap_perf[st][metric]['unc'], c=c1, ls='-')
    axs[2,1].scatter(bin_centers, umap_perf[st][metric]['unc'], c=c1, s=dotsize)
    axs[2,1].plot(bin_centers, colors_perf['biases'], c=c4, ls='-')
    axs[2,1].scatter(bin_centers, colors_perf['biases'], c=c4, s=dotsize)
    axs[2,1].set_ylim(0,bias_ymax)
    #axs[2,1].set_xlim(0,None)
    axs[2,1].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)




    fig.subplots_adjust(top=0.95)
    fig.legend(bbox_to_anchor=(0.03, 1.02, 1., .102), loc='center',
                        ncols=2, borderaxespad=0., fontsize=20)
    fig.tight_layout()

    if filename!=None:
        fig.savefig(filename, bbox_inches='tight')


def sandplot_rachel(uncs, unc_sources, labels, cs, ylims=None, filename=None):
    '''
    '''
    n_uncs = len(unc_sources)

    fig = plt.figure(figsize=(12, 10))

    outer = fig.add_gridspec(
        nrows=3,
        ncols=2,
        wspace=0.1,
        hspace=0.3
    )

    all_axes = []

    metrics = ['f_outlier', 'nmad', 'bias']
    ylabels = [
        'Contribution to '+r'$f_\mathrm{outlier}$'+' Uncertainty',
        'Contribution to '+r'$\sigma_\mathrm{NMAD}$'+' Uncertainty',
        'Contribution to Bias Uncertainty'
    ]
    alpha=0.5

    for row in range(3):
        for col in range(2):

            inner = outer[row, col].subgridspec(
                nrows=n_uncs,
                ncols=1,
                hspace=0.0
            )

            panel_axes = [fig.add_subplot(inner[0])]
            panel_axes += [
                fig.add_subplot(inner[i], sharex=panel_axes[0])
                for i in range(1, n_uncs)
            ]
            #ticks
            for ax in panel_axes:
                ax.tick_params(axis='both', labelsize=18, length=6, width=1.5, 
                                bottom=True,top=False,left=True, right=False,
                                direction='out')
            # Hide x tick labels on upper axes within the panel
            for ax in panel_axes[:-1]:
                ax.tick_params(labelbottom=False)

            # Only bottom row gets x-labels
            if row == 2:
                panel_axes[-1].set_xlabel(r'$z_\mathrm{LePHARE}$', fontsize=25)

            # One y-label per panel, left column only
            if col == 0:
                panel_axes[n_uncs // 2].set_ylabel(
                    ylabels[row],
                    labelpad=15
                )

            all_axes.append(panel_axes)

    # Example data
    x = np.linspace(0, 10, 500)

    for panel_axes in all_axes:
        for i, ax in enumerate(panel_axes):
            ax.plot(x, np.sin(x + i))

    #axs[0,0].set_title('LePHARE-trained', fontsize=25)
    #axs[0,1].set_title('spec-'+r'$z$'+'-trained', fontsize=25)
    fig.subplots_adjust(top=0.95)
    fig.legend(bbox_to_anchor=(0.03, 0.98, 1., .102), loc='center',
                        ncols=3, borderaxespad=0., fontsize=22)
    #fig.tight_layout()

    plt.show()

# END