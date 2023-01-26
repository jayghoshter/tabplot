#!/usr/bin/env python3

from tabplot import BarPlot

BarPlot(
    files=["./bar_a.csv", "./bar_b.csv"],
    labels=["A", "B"],
    bar_width=0.4,
    legend_loc="upper right",
    xlabel="Category",
    ylabel="Value",
    title="Some Bar Plot",
).read().draw().save("barplot.pdf")
