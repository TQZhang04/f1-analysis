import pandas as pd
import numpy as np

# datavis
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from scipy import stats
import os
import io

# function to plot distribution differences
def plot_missing_diff(df, miss, col, barmode=''):
    """Plots difference in distribution for a categorical column between missing
    and present data in another column.
    
    df: data
    miss: name of column with missing data
    col: name of column to plot distribution of
    """
    mode = (barmode if barmode
            else 'overlay' if df[col].dtype in ['float', 'int'] 
            else 'group')
    return px.histogram(df.assign(missing=df[miss].isna()),
                x=col,
                color='missing',
                histnorm='probability density',
                title=f'Distribution of {col} between Missing and Present {miss}',
                barmode=mode,
                color_discrete_map={
                    False: '#ff7f0e',
                    True: '#1f77b4'
                })
    
# tvd function from lecture 8, slightly modified to work more generally
def tvd_of_groups(df, groups, cats):
    '''groups: the binary column (e.g. married vs. unmarried).
       cats: the categorical column (e.g. employment status).
    '''
    cnts = df.pivot_table(index=cats, columns=groups, aggfunc='size')
    # Normalize each column.
    distr = cnts / cnts.sum()
    # Compute and return the TVD.
    return (distr.iloc[:,0] - distr.iloc[:,1]).abs().sum() / 2

def missingness_perm_test_cat(df, cat_col, missing_col, N=1000, showplot=False):
    """Runs a missingness permutation test on the categorical column given
    """
    # make a mini dataframe with only columns of interest
    df_mini = df[[missing_col, cat_col]].copy()
    df_mini[missing_col] = df_mini[missing_col].isna()
    df_mini.head()

    # permutation test
    tvd = lambda x, y: np.abs(x - y).sum() / 2

    # Step 1: Calculate observed tvds
    obs_tvd = tvd_of_groups(df_mini, cat_col, missing_col)

    # Step 2: generate N simulated tvds
    sim_tvds=[]
    for _ in range(N):
        df_mini['shuffled_cat'] = np.random.permutation(df_mini[cat_col])
        sim_tvd = tvd_of_groups(df_mini, 'shuffled_cat', missing_col)
        sim_tvds.append(sim_tvd)

    # Step 3: calculate p-value, extreme direction is to the right (greater than)
    p_val = (np.array(sim_tvds) >= obs_tvd).mean()
    if showplot:
        fig = px.histogram(sim_tvds, nbins=20)
        fig.add_vline(obs_tvd)
        fig.show()
    return p_val