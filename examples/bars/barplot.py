#!/usr/bin/env python3

from tabplot import BarPlot

BarPlot(
    files = ['./bar_a.csv', './bar_b.csv'],
    labels = ['A', 'B'],
    bar_width = 0.4,
    legend = ('upper right', 1, 1),
    xlabel = 'Category',
    ylabel = 'Value',
    title = 'Some Bar Plot'
).read().draw().save('barplot.pdf')
