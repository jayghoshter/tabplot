import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from tabplot.utils import make_iterable
from cycler import cycler

# TODO: GPR

def parametric_line(ax, slope, intercept, xlog=False, ylog=False, **kwargs):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())

    if xlog: 
        x_vals = np.log10(x_vals)

    # if self.ylog: 
    #     y_vals = np.log10(y_vals)

    y_vals = intercept + slope * x_vals

    if xlog: 
        x_vals = np.power(10,x_vals)
    if ylog: 
        y_vals = np.power(10,y_vals)

    return ax.plot(x_vals, y_vals, **kwargs)

def fit_lines(ax, xs, ys, xlog=False, ylog=False, labels=None, zorders=None, **kwargs):
    lines = []

    labels = make_iterable(labels, '', len(xs), return_list=True)
    zorders = make_iterable(zorders, -1, len(xs), return_list=True)

    for key in kwargs: 
        kwargs[key] = make_iterable(kwargs[key], None, len(xs), return_list=True)

    custom_cycler = cycler(**kwargs)
    ax.set_prop_cycle(custom_cycler)

    for x,y,label,zorder in zip(xs, ys, labels, zorders): 
        X = np.array(x)
        Y = np.array(y)

        ordering = X.argsort()
        X = X[ordering].reshape(-1,1)
        Y = Y[ordering]

        if xlog: 
            X = np.log10(X)
        if ylog: 
            Y = np.log10(Y)

        model = LinearRegression()
        model.fit(X, Y)

        score = model.score(X,Y)
        line = parametric_line(ax, model.coef_, model.intercept_ , label=label, zorder=zorder)
        print(f"R2={score} | m={model.coef_} | c={model.intercept_}")
        lines.extend(line)

    return lines

def extrapolate(ax, xs, ys, kind='linear'):
    xlim = ax.get_xlim()
    xsmooth = np.linspace(xlim[0], xlim[1], 250) 
    for x,y in zip(xs, ys): 
        ysmooth = interp1d(x, y, kind=kind, fill_value='extrapolate')(xsmooth)
        ax.plot(xsmooth, ysmooth, label=f"{kind} extrapolation")
