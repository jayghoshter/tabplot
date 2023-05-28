#!/usr/bin/env python3

"""
An example with no file read. plot.xs and plot.ys are set manually.
"""

from tabplot import Plot
import numpy as np

plot = Plot()
plot.figsize = (4.0, 4.0)
x = np.linspace(-10, 10, 100)
y = (10 * x) ** 2
plot.xs = [x]
plot.ys = [y]
plot.labels = [r'$(10x)^2$']
plot.show_legend = True
plot.draw().save('test.png')
