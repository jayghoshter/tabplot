from itertools import cycle, islice, repeat
import matplotlib as mpl
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import numexpr as ne

from typing import Iterable

def make_iterable(obj, default_value, default_length, return_list=True) -> Iterable:
    """
    For a given object, 
        - if it is false-like, return [ default_value ] * default_length
        - if it is an iterable, cycle + islice to the correct length
        - if it is not an iterable, repeat + islice to the correct length
    """
    if not obj:
        obj = iter([default_value] * default_length)
    elif hasattr(obj, '__iter__') and not isinstance(obj, str):
        obj = islice(cycle(obj), default_length)
    else:
        obj = islice(repeat(obj), default_length)

    if return_list:
        return list(obj)
    else:
        return obj

def readfile(data_path, delimiter=None, header=False):
    if delimiter is None:
        with open(data_path, newline='') as csvfile:
            if ',' in csvfile.readline():
                delimiter = ','

    skiprows = 1 if header else 0

    # Whitespace delimiter
    if delimiter is None: 
        return np.loadtxt(data_path, dtype=object, skiprows=skiprows).T
    else:
        # Comma or specified
        return np.loadtxt(data_path, dtype=object, delimiter=delimiter, skiprows=skiprows).T 

def normalize(data:np.ndarray, refValue=None):
    """ Normalize array to either max value or given refValue
    """
    if not refValue:
        refValue = data.max()
    return data / refValue

def scale_axis(vec:np.ndarray, scale_factor_or_file):
    """ Scale either the x or y axis of a given plot line
    """
    if isinstance(scale_factor_or_file, float) or isinstance(scale_factor_or_file, int) or isinstance(scale_factor_or_file, np.ndarray):
        return vec * scale_factor_or_file
    elif Path(scale_factor_or_file).expanduser().exists():
        scale_factor = np.loadtxt(scale_factor_or_file)
        return vec * scale_factor
    else: 
        print(f"No such scaling factor or data file: {scale_factor_or_file}")
        return vec

def cmap_colors_to_hex(cmap_colors): 
    # USAGE: cmap_colors_to_hex(plt.cm.tab10.colors)
    return list(map(lambda x: mpl.colors.rgb2hex(x), cmap_colors))

def xsort(x:np.ndarray, y:np.ndarray):
    ordering = x.argsort()
    x = x[ordering]
    y = y[ordering]
    return x, y

def trim(x:np.ndarray, y:np.ndarray, condition:str):
    mask = ne.evaluate(condition)
    indices = np.where(mask)
    x = x[indices]
    y = y[indices]
    return x, y

def strim(xs:list[np.ndarray], ys:list[np.ndarray], condition:str):
    xs, ys = zip(*map(trim, xs, ys, repeat(condition)))
    return xs, ys

def xssort(xs:list[np.ndarray], ys:list[np.ndarray]):
    xs, ys = zip(*map(xsort, xs, ys))
    return xs, ys

def smoothen_xys(xs, ys, order=3, npoints=250):
    new_xs = []
    new_ys = []
    for x, y in zip(xs, ys): 
        xsmooth = np.linspace(min(x), max(x), npoints)
        x,y = xsort(x,y)
        spl = make_interp_spline(x, y, k=order)  # type: BSpline
        ysmooth = spl(xsmooth)
        new_xs.append(xsmooth)
        new_ys.append(ysmooth) 
    return new_xs, new_ys
