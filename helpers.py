import pandas as pd
import numpy as np

import itertools as it
import functools as ft

def map_list(*args, **kwargs):
    '''Returns an evaluated map as a list'''
    return list(map(*args, **kwargs))

def filter_list(*args, **kwargs):
    '''Returns an evaluated filter as a list'''
    return list(filter(*args, **kwargs))

def display_hist(x, num_bins = 8, zeros_as_blank = False):
    '''Returns a histogram as a unicode text string, e.g. '▁▂▄█▆▃▁▁'
    
    Inspired by the `precis` function from the Statistical Rethinking R package
    by Richard McElreath. This function will calculate a histogram and then
    returns a string displaying the histogram in unicode characters. It uses the
    LOWER BLOCK group like "2584 ▄ LOWER HALF BLOCK". 
    
    After I published this I was alerted to the correct term for this type of text
    plot: spark lines. There is a python package by @RedKrieg that is much more 
    robust for turning a sequence into a spark line called pysparklines. And the
    original(?) terminal package form @holman called spark:
    * pysparklines: https://github.com/RedKrieg/pysparklines
    * spark: https://github.com/holman/spark
    
    Parameters
    ----------
    x : numpy.array 
        The vector of values to compute the histogram for
    num_bins : int or list of float
        The number of characters to print out. Can pass custom bin edges to 
        `np.histogram` as well.
    zeros_as_blank : bool
        Should buckets with 0 observations be a blank space, False would still
        show a one eight block if there are no observations.
        
    Returns
    -------
    unicode_str : str
        The histogram str to be displayed
        
    Examples
    --------
    >>> display_hist(np.random.uniform(size = 1000))
    '▇▇▇▇▇▆▇█'
    >>> display_hist(np.random.normal(size = 1000))
    '▁▂▄█▆▃▁▁'
    >>> display_hist(np.abs(np.random.normal(size = 1000)))
    '█▇▅▃▂▁▁▁'
    >>> display_hist(np.power(np.random.normal(size = 1000), 2))
    '█▂▁▁▁▁▁▁'
    >>> display_hist(np.hstack([np.repeat(0, 900), np.repeat(10, 100)]), zeros_as_blank = True)
    '█      ▁'
    >>> display_hist(np.hstack([np.repeat(0, 900), np.repeat(10, 100)]))
    '█▁▁▁▁▁▁▁'
    >>> display_hist(np.hstack([np.random.normal(size = 1000), 
                                np.random.normal(loc = 3, scale = 0.5, size = 1000)]), 
                     num_bins = 16)
    '▁▁▂▂▃▅▄▄▂▁▃▆█▄▁▁'
    
    References
    ----------
    The unicode code charts: https://www.unicode.org/Public/UCD/latest/charts/CodeCharts.pdf
    '''
    
    ## Get bin counts as a pct of total obs
    hist_counts, bin_edges = np.histogram(x, bins = num_bins)
    x_total = x.shape[0]
    pct_counts = hist_counts / x_total
    ## scale the percentages by the max pct and 0, then convert to the index
    ## of the appropriate unicode string in unicode_list
    max_pct = np.max(pct_counts)
    bin_labels = np.floor(pct_counts * (8 - 1) / max_pct).astype('int')
    ## adjust zeros to blank space index
    if zeros_as_blank:
        zero_ind = pct_counts == 0.0
        bin_labels[zero_ind] = 8
        
    unicode_list = ['\u2581', '\u2582', '\u2583', '\u2584',
                    '\u2585', '\u2586', '\u2587', '\u2588', ' ']
    unicode_labels = [unicode_list[l] for l in bin_labels]
    unicode_str = ft.reduce(lambda x, y: x + y, unicode_labels)
    return unicode_str

class BitmapEncoder():
    
    def __init__(self, categories = 'auto', handle_unknown = 'error', max_categories = None):
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        
    def fit(self, X, y = None):
        if not isinstance(X, (pd.core.series.Series, np.ndarray)):
            raise TypeError('X is not a numpy array or Series')
        
        try:
            self.name = X.name
        except:
            self.name = 'category'
        
        self.x_values = np.unique(X)
        self.num_values = len(self.x_values)
        self.num_bits = np.ceil(np.log2(self.num_values)).astype('int')
        bit_combos = list(it.product(*it.repeat([1, 0], self.num_bits)))
        self.bitmap = dict(zip(self.x_values, bit_combos))
        self.bitmap_df = pd.DataFrame(bit_combos, columns = [f'{self.name}_{i}' for i in range(self.num_bits)]).iloc[:self.num_values]
        self.bitmap_df.insert(0, self.name, self.x_values)
        X_transformed = np.array([self.bitmap[x] for x in X])
        return X_transformed
        
    def fit_df(self, X, y = None):
        X_transformed = self.fit(X)
        bit_df = pd.DataFrame(X_transformed, columns = [f'{self.name}_{i}' for i in range(self.num_bits)])
        bit_df.insert(0, self.name, X)
        return bit_df
    
    def transform(self, X):
        
        x_vals = np.unique(X)
        if any(x_vals not in self.x_values):
            raise AssertionError('X contains new categories not found in fitted data')
        
        X_transformed = np.array([self.bitmap[x] for x in X])
        return X_transformed
    
    def transform_df(self, X):
        X_transformed = self.transform(X)
        bit_df = pd.DataFrame(X_transformed, columns = [f'{self.name}_{i}' for i in range(self.num_bits)])
        bit_df.insert(0, self.name, X)
        return bit_df
        
        
def bitmap_encoding(cat_col):
    '''Convert a string column into bit labels
    
    Each label will receive a unique bit encoding of 0's and 1's that
    Can identify that label. This helps trees and deep learning libraries
    narrow down to specific categories faster.'''
    col_name = cat_col.name
    cat_str = cat_col.copy().astype('str')
    cat_str = cat_str.fillna('Missing')
    cat_values = cat_str.unique()
    num_values = len(cat_values)
    cat_df = pd.DataFrame({col_name: cat_values, f'{col_name}_index': np.arange(num_values)})
    num_bits = round(np.ceil(np.log2(num_values)))
    bit_combos = it.product(*it.repeat([0, 1], num_bits))
    bit_df = pd.DataFrame(bit_combos, columns = [f'bit_{i}' for i in range(num_bits)])
    bit_df[f'{col_name}_index'] = np.arange(bit_df.shape[0])
    bit_df = pd.merge(cat_df, bit_df, on = f'{col_name}_index', how = 'inner')
    return bit_df


retro_palette = np.array(
    ['#D73F27', '#E8713D', '#EAA86C', '#F2DDB3',
    '#81A9A0', '#2A8B99', '#1E6E8D', '#142B54']
)

def retro_palette_scaled(n = 1): 
    '''Uses the predefined retro palette to return color scales of less than 8 values'''
    if n == 1:
        scaled_palette = retro_palette[0]
    if n >= 8:
        scaled_palette = retro_palette
    else:
        pos = np.round(np.linspace(0, 7, n, endpoint=True)).astype('int')
        scaled_palette = retro_palette[pos]
    
    return scaled_palette

def theme_custom(**kwargs):
    custom_theme_args = dict(
        dpi = 150, 
        figure_size = (7, 4), 
        plot_background=element_rect(fill = '#F5F5F5'), panel_background=element_rect(fill = '#F5F5F5'))   
    ## use custom theme args first so the given options will override them
    theme_args = {**custom_theme_args, **kwargs}
    
    return theme(**theme_args)