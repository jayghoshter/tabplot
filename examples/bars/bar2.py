#!/usr/bin/env python3

from tabplot import BarPlot

BarPlot(style='science', bar_width=0.5, xticklabels_rotation=45).read(files=['./commits-by-date.csv'], columns=(-999,1), xticklabels_column=0).draw().show()

(
    BarPlot(
        style='science', 
        bar_width=0.5, 
        xticklabels_rotation=90,
        strip_xticklabels=True,)
    .read(
        files=['./commits-by-date-2.csv'], 
        columns=(-999,0), 
        xticklabels_column=1)
    .setup()
    .setrc({'xtick.labelsize': 8})
    .draw()
    .show()
    )
