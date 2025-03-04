from itertools import cycle, islice, repeat
import matplotlib as mpl
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import numexpr as ne
from fuzzywuzzy import process, fuzz

from typing import Iterable

def find_files(pattern:str, dir='.', recursive=False, ignore_files:list|None=None, ignore_dirs:list|None=None):
    # return [ p for p in Path('.').rglob('*') if p.is_file() and STATEDIR not in p.parents]
    if ignore_dirs is not None:
        ignore_dirs = [Path(dir) for dir in ignore_dirs]
    else: 
        ignore_dirs = []

    if ignore_files is not None:
        ignore_files = [Path(f) for f in ignore_files]
    else: 
        ignore_files = []

    if recursive: 
        files = [ p for p in Path(dir).rglob(pattern) if p.is_file() ]
    else: 
        files = [ p for p in Path(dir).glob(pattern) if p.is_file() ]

    for dir in ignore_dirs:
        files = list(filter(lambda f: dir not in f.parents, files))

    for ignore_file in ignore_files:
        files = list(filter(lambda f: ignore_file != str(f), files))

    return files

def make_iterable(obj, default_value, default_length, return_list=True) -> Iterable:
    """
    For a given object,
        - if it is false-like, return [ default_value ] * default_length
        - if it is an iterable, cycle + islice to the correct length
        - if it is not an iterable, repeat + islice to the correct length
    """
    if not obj:
        obj = iter([default_value] * default_length)
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        obj = islice(cycle(obj), default_length)
    else:
        obj = islice(repeat(obj), default_length)

    if return_list:
        return list(obj)
    else:
        return obj


def readfile(data_path, delimiter=None, header=False, dtype=object):
    if delimiter is None:
        with open(data_path, newline="") as csvfile:
            if "," in csvfile.readline():
                delimiter = ","

    skiprows = 1 if header else 0

    # Whitespace delimiter
    if delimiter is None:
        return np.loadtxt(data_path, dtype=dtype, skiprows=skiprows).T
    else:
        # Comma or specified
        return np.loadtxt(
            data_path, dtype=dtype, delimiter=delimiter, skiprows=skiprows
        ).T

def readheader(data_path, delimiter=None):
    headers = []
    import csv
    from io import StringIO
    with open(data_path, newline="") as csvfile:
        headerline = csvfile.readline()
        if ',' in headerline: 
            header_io = StringIO(headerline)
            headers.append(csv.reader(header_io, delimiter=',').__next__())
    return headers

def fuzzy_str_to_idx(target:str|None, itemlist:list[str], sentinel_idx:int=-999):
    if target is None:
        return sentinel_idx
    itemdict = {idx: el for idx, el in enumerate(itemlist)}
    result = process.extractOne(target, itemdict, scorer=fuzz.token_set_ratio)
    if result:
        return result[2] # Returns the index
    return None

def normalize(data: np.ndarray, refValue=None):
    """Normalize array to either max value or given refValue"""
    if not refValue:
        refValue = data.max()
    return data / refValue


def scale_axis(vec: np.ndarray, scale_factor_or_file):
    """Scale either the x or y axis of a given plot line"""
    if (
        isinstance(scale_factor_or_file, float)
        or isinstance(scale_factor_or_file, int)
        or isinstance(scale_factor_or_file, np.ndarray)
    ):
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


def xsort(x: np.ndarray, y: np.ndarray):
    ordering = x.argsort()
    x = x[ordering]
    y = y[ordering]
    return x, y


def trim(x: np.ndarray, y: np.ndarray, condition: str):
    mask = ne.evaluate(condition)
    indices = np.where(mask)
    x = x[indices]
    y = y[indices]
    return x, y


def strim(xs: list[np.ndarray], ys: list[np.ndarray], condition: str):
    xs, ys = zip(*map(trim, xs, ys, repeat(condition)))
    return xs, ys


def xssort(xs: list[np.ndarray], ys: list[np.ndarray]):
    xs, ys = zip(*map(xsort, xs, ys))
    return xs, ys


def smoothen_xys(xs, ys, order=3, npoints=250):
    new_xs = []
    new_ys = []
    for x, y in zip(xs, ys):
        xsmooth = np.linspace(min(x), max(x), npoints)
        x, y = xsort(x, y)
        spl = make_interp_spline(x, y, k=order)  # type: BSpline
        ysmooth = spl(xsmooth)
        new_xs.append(xsmooth)
        new_ys.append(ysmooth)
    return new_xs, new_ys

def get_colors_from(cmap='tab10', discretization=2):
    cmap = mpl.cm.get_cmap(name=cmap)
    if "colors" in cmap.__dict__:
        # Discrete colormap
        colors = cmap.colors
    else:
        # Continuous colormap
        # WARNING: doesn't handle this perfectly
        colors = [
            cmap(1.0 * i / (discretization - 1)) for i in range(discretization)
        ]
    return colors
