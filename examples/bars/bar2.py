#!/usr/bin/env python3

from tabplot import BarPlot

BarPlot(style='science', bar_width=1).read(files=['./commits-by-date.csv'], columns=(-1,1), xticklabels_column=0).draw().show()
