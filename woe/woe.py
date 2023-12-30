import pandas as pd
import numpy as np

## from plotnine.data import mpg
## mpg['fuel_effecient'] = 1.0 * (mpg['cty'] > 25)

def bin_data(col, bins = 8, **kwargs):
    ''' 
    Converts numeric columns to column bins
    
    :param bins: The number of bins to return. If the number of bins is more than
                 the number of unique values in the series then we just return
                 the values themselves
    :param kwargs: Arguments to `pd.cut`
    '''
    default_params = dict(
        include_lowest = True,
        duplicates = 'drop'
    )
    ## overwrite defaults with kwargs
    cut_args = {**default_params, **kwargs}
    num_unique = col.nunique()
    ## So rather than group the data if the number of unique values is less than
    ## the desired number of bins...Just return the column as is and we will group
    ## by the actual values when we calculate our summary
    if num_unique < bins:
        bin_labels = col
    ## If not then we need to group the values into bins using cut
    else:
        num_bins = bins
        bin_labels = pd.cut(col, bins = num_bins, **cut_args)
    
    return bin_labels


def trim_categories(col, bins = 8):
    '''Group the smallest categories into an `_other_` category'''
    cat_counts = col.value_counts(dropna = False)
    ## if the number of categories is already less than the max bins then just
    ## use the raw values
    if bins > cat_counts.shape[0]:
        return col
    
    ## We know value_counts returns a sorted array
    small_cats = cat_counts.iloc[(bins - 1):].index.to_list()
    default_cats = col.replace(to_replace = small_cats, value = '_other_')
    
    return default_cats


def calc_woe(df, feature_col, dv_col, min_obs = 1, **bin_args):
    '''
    Calculate the WOE and IVs for the categories of the feature
    
    :param df: The dataframe containing the columns to use
    :param feature_col: The name of the column that contains the feature values
    :param dv_col: The name of the column that contains the dependent variable
    :param min_obs: The amount to add to each numerator and denominator when
                    calculating the percent of overall goods and bads to avoid
                    taking logs of 0
    '''
    df_ = df[[feature_col, dv_col]].copy()
    dv_levels = df_[dv_col].unique()
    
    if len(dv_levels) != 2:
        raise(f'Need only 2 levels for {dv_col}')
    
    num_bads = np.sum(df_[dv_col] == dv_levels[0])
    num_goods = np.sum(df_[dv_col] == dv_levels[1])
    
    if str(df_[feature_col].dtype) in ['string', 'category']:
        df_[feature_col + '_bins'] = trim_categories(df_[feature_col], bin_args['bins'])
    else:
        df_[feature_col + '_bins'] = bin_data(df_[feature_col], **bin_args)
    
    df_counts = (
        df_.
        groupby([feature_col + '_bins'], dropna = False).
        apply(lambda df: pd.Series(dict(
            num_obs = df.shape[0],
            num_bads = np.sum(df[dv_col] == dv_levels[0]),
            num_goods = np.sum(df[dv_col] == dv_levels[1])
        ))).
        reset_index()
    )
    
    df_counts['pct_goods'] = (df_counts['num_goods'] + min_obs) / (num_goods + min_obs)
    df_counts['pct_bads'] = (df_counts['num_bads'] + min_obs) / (num_bads + min_obs)
    df_counts['woe'] = np.log(df_counts['pct_goods'] / df_counts['pct_bads'])
    df_counts['iv'] = df_counts['woe'] * (df_counts['pct_goods'] - df_counts['pct_bads'])
    
    return df_counts


def calc_woe_cont(df, feature_col, dv_col, min_obs = 1, **bin_args):
    '''
    Calculate the WOE and IVs for the categories of the feature for numeric
    dependent values
    
    :param df: The dataframe containing the columns to use
    :param feature_col: The name of the column that contains the feature values
    :param dv_col: The name of the column that contains the dependent variable
    :param min_obs: The amount to add to each numerator and denominator when
                    calculating the percent of overall goods and bads to avoid
                    taking logs of 0
    '''
    df_ = df[[feature_col, dv_col]].copy()
    ## Rather than worry about handling negative values I'll just make every 
    ## DV value positive. WOE & IV shouldn't change based on the scale of the DV 
    ## anyway so this should be fine
    dv_col_ = dv_col + '_scaled'
    df_[dv_col_] = df[dv_col] - df[dv_col].min()
    
    num_obs = df_.shape[0]
    num_dv = np.sum(df_[dv_col_])
    
    ## We could implement functionality to reduce these categorical variables to
    ## a max number of values....
    if str(df_[feature_col].dtype) in ['string', 'category']:
        df_[feature_col + '_bins'] = trim_categories(df_[feature_col], bin_args['bins'])
    else:
        df_[feature_col + '_bins'] = bin_data(df_[feature_col], **bin_args)
    
    df_counts = (
        df_.
        groupby([feature_col + '_bins'], dropna = False).
        apply(lambda df: pd.Series(dict(
            num_obs = df.shape[0],
            num_dv = np.sum(df[dv_col_])
        ))).
        reset_index()
    )
    
    df_counts['pct_dv'] = (df_counts['num_dv'] + min_obs) / (num_dv + min_obs)
    df_counts['pct_baseline'] = (df_counts['num_obs'] + min_obs) / (num_obs + min_obs)
    df_counts['woe'] = np.log(df_counts['pct_dv'] / df_counts['pct_baseline'])
    df_counts['iv'] = df_counts['woe'] * (df_counts['pct_dv'] - df_counts['pct_baseline'])
    
    return df_counts


def calc_iv(df, feature_col, dv_col, binary = True, min_obs = 1, **bin_args):
    '''
    Calculate Information Value for the feature
    
    :param df: The dataframe containing the columns to use
    :param feature_col: The name of the column that contains the feature values
    :param dv_col: The name of the column that contains the dependent variable
    '''
    if binary:
        woe_df = calc_woe(df, feature_col, dv_col, min_obs, **bin_args)
    else:
        woe_df = calc_woe_cont(df, feature_col, dv_col, min_obs, **bin_args)
        
    return np.sum(woe_df['iv'])

    

    