#!/usr/bin/env python3

from tabplot import Plot

# Initialize Plot object
plot = Plot()

# Set files and plot attributes/properties (matplotlib)
plot.files = ["./line_a.tsv", "./line_b.tsv"]  # List of files to read
plot.labels = ["A", "B"]  # labels per file/line

# Plot properties can be their primitive values (str/float) or a list of them.
# If primitives are used, the same value is applied to all plot lines
# Otherwise, the property for each plot line is taken from the given list
plot.linewidths = 2  # or [2,2] etc

plot.legend_loc = "upper left"

# Read, draw and save the plot
plot.read()
plot.draw()
plot.save("simple-1.pdf")
plot.close()
# # OR: use operator chaining to do the same
# plot.read().draw().save('plot.pdf')

# Initialize with kwargs
Plot(
    files=["./line_b.tsv", "line_a.tsv"],
    labels=["A", "B"],
    markers="D",
    legend_fontsize=20,
).read().draw().save("simple-2.pdf").close()

# Initialize with dict
attrs = {
    "files": ["./line_a.tsv"],
    "labels": ["The one true curve"],
    "linestyles": "dashdot",
    "linewidths": 2,
    "legend_loc": "upper left",
}
Plot(**attrs).read().draw().save("simple-3.pdf").close()
