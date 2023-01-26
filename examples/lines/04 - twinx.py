#!/usr/bin/env python3

from tabplot import Plot

Plot(
    files=["./line_a.tsv"],
    twinx=["./line_b.tsv"],
    labels=["A", "B"],
    xlabel="X",
    ylabel="Y",
    y2label="Y2",
    linewidths=2,
).read().draw().show()
