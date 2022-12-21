from itertools import cycle, islice, repeat
from subprocess import run
import matplotlib as mpl

def make_iterable(obj, default_value, default_length, return_list=True):
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

def readfile(data_path, columns=[0,1], header=False, xticksColumn=None, xticklabelsColumn=None):
    """ Read x-y CSV-style files
    """
    if ':' in data_path:
        run(['scp', '-rC', data_path, '/tmp/plotting.csv'])
        data_path = '/tmp/plotting.csv'

    x = []
    y = []
    xticks = []
    xticklabels = []
    # columns = [0, 1]
    delimiter = ' '
    with open(data_path, newline='') as csvfile:
        if ',' in csvfile.readline():
            delimiter = ','
    with open(data_path, newline='') as infile:
        # data = list(csv.reader(infile))
        if header:
            print(infile.readline())
        for line in infile:
            data_line = line.strip().split(delimiter)
            data_line = list(filter(None, data_line))
            if (data_line != []):
                if len(data_line) == 1:
                    y.append(float(data_line[0]))
                elif len(columns) == 1: 
                    y.append(float(data_line[columns[0]]))
                else:
                    x.append(float(data_line[columns[0]]))
                    y.append(float(data_line[columns[1]]))
                # if columns[0] != -1:
                #     x.append(float(data_line[columns[0]]))
                # y.append(float(data_line[columns[1]]))
                if xticksColumn is not None: 
                    xticks.append(float(data_line[xticksColumn]))
                if xticklabelsColumn is not None:
                    xticklabels.append(data_line[xticklabelsColumn])

    return x, y, xticks, xticklabels

def normalize(data, refValue=None):
    """ Normalize array to either max value or given refValue
    """
    if refValue == 'self':
        print("No reference value for normalization provided: Using max point of curve.")
        refValue = max(data)
    return [ x/float(refValue) for x in data ]

def scale_axis(vec:list, scale_factor_or_file):
    """ Scale either the x or y axis of a given plot line
    """
    if Path(scale_factor_or_file).expanduser().exists():
        scale_factors = readArray(scale_factor_or_file)
        vec = [ v*s for v,s in zip(vec, scale_factors)]
    else: 
        scale_factor = float(scale_factor_or_file)
        vec = [ v * scale_factor for v in vec ]
    return vec

def cmap_colors_to_hex(cmap_colors): 
    # USAGE: cmap_colors_to_hex(plt.cm.tab10.colors)
    return list(map(lambda x: mpl.colors.rgb2hex(x), cmap_colors))
