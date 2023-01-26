#!/usr/bin/env python3

from tabplot import Histogram

Histogram().read(files=["hist1.csv"]).draw().save("hist1.pdf")

Histogram().read(files=["hist1.csv", "hist2.csv"]).scale(y=1000000).draw().save(
    "hist_both.pdf"
)

Histogram(stacked=True).read(files=["hist1.csv", "hist2.csv"]).scale(
    y=1000000
).draw().save("hist_both_stacked.pdf")
