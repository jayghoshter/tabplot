#!/usr/bin/env python3

from tabplot import Plot

plot = Plot()
plot.files = ["./line_a.tsv"]
plot.labels = ["A"]
plot.linewidths = 2
plot.legend_loc = "upper left"
plot.read().draw().save("multi-1.pdf")

plot.files = ["./line_b.tsv"]
plot.labels = ["B"]
plot.line_color_indices = 1
# NOTE: When clean=False, any previous rendered plot remains and we plot over it
plot.read().draw(clean=False).save("multi-2.pdf")

plot.files = ["./line_a.tsv", "./line_b.tsv"]
plot.labels = ["A", "B", "C"]
plot.line_color_indices = []  # reset line_color_indices
plot.read().draw().save("multi-3.pdf")
