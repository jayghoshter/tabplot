import numpy as np
from sklearn.linear_model import LinearRegression

def parametric_line(ax, slope, intercept, xlog=False, ylog=False):
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

    ax.plot(x_vals, y_vals, c='gray')


def fit_line(ax, xs, ys, xlog=False, ylog=False):
    for x,y in zip(xs, ys): 
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
        parametric_line(ax, model.coef_, model.intercept_)
        print(f"R2={score} | m={model.coef_} | c={model.intercept_}")
