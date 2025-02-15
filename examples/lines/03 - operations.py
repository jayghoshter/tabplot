#!/usr/bin/env python3

from tabplot import Plot

plot = Plot()
plot.files = ["./line_a.tsv", "./line_a.tsv", "./line_a.tsv"]
plot.labels = ["A", "A-scaled-x", "A-scaled-y"]
plot.load().multiscale(x=[1, 1.2, 1], y=[1, 1, 1.2]).draw().save(
    "operations-1.pdf"
).close()


plot = (
    Plot(
        files=["./line_a.tsv"],
    )
    .load()
    .normalize_xy()
    .trim("x<0.4")
    .draw()
    .save("operations-2.pdf")
)

plot.files = ["./line_b.tsv"]
plot.colormap = "Accent"
plot.fill = 0.4
plot.load().normalize_xy().draw().save("operations-3.pdf").close()


# hlines and fit_lines
plot = (
    Plot(files=["./line_a.tsv", "./line_b.tsv"], labels=["A", "B"])
    .load()
    .normalize_xy()
    .multiscale(y=[1.0, 2.0])
    .draw()
)
plot.fit_lines(labels=["one", "two"], color=["magenta", "green"], linewidth=0.5)
plot.hlines(
    [0.6, 0.8], colors=["gray", "r"], alpha=0.6, zorder=-1, ls="dashdot", lw=0.5
)
plot.vlines([0.6, 0.8], colors="black", alpha=0.6, zorder=-1, ls="dashdot", lw=0.5)
plot.save("operations-4.pdf")
