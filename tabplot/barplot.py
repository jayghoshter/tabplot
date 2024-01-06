from tabplot import Plot
import matplotlib.pyplot as plt
import numpy as np
from math import isnan

class BarPlot(Plot):
    bar_width: float

    def __init__(self, **kwargs) -> None:
        self.bar_width = 0.25
        self.strip_xticklabels: bool = True
        super().__init__(**kwargs)

    def _plot_data(self, ax, xs, ys, labels, zorders):
        lines = []
        bar_count = 0
        indices = list(range(1, len(ys[0]) + 1))
        num_files = len(ys)

        for x, y, label in zip(xs, ys, labels):
            y = [ 0.0 if isnan(iy) else iy for iy in y ]
            width = self.bar_width
            shiftwidth = (num_files - 1) * width / 2.0
            position = [bar_count * width - shiftwidth + i for i in indices]
            line = ax.bar(position, y, width=width, label=label)
            lines.append(line)
            bar_count = bar_count + 1
            # if args['xticks_column'] is not None:
            #     plt.xticks(indices,xticks)
        return lines

    def _setup_ticks(
        self, xticks=None, yticks=None, xtick_labels=None, ytick_labels=None
    ):
        if xticks is None:
            xticks = self.xticks

        if yticks is None:
            yticks = self.yticks

        if xtick_labels is None:
            xtick_labels = self.xtick_labels

        if ytick_labels is None:
            ytick_labels = self.ytick_labels

        xtx, xtl = plt.xticks()

        # Use indices unless xticks/labels are explicitly provided
        indices = list(range(1, len(self.ys[0]) + 1))

        if len(xticks):
            xtx = xticks
        else: 
            xtx = indices

        if len(xtick_labels):
            xtl = xtick_labels
        else: 
            xtl = [ str(x) for x in xtx ]

        # Strip xticklabels if values are 0
        if self.strip_xticklabels:
            ys = np.array(self.ys)
            for i in range(len(xtl)):
                if not any(ys[:,i]):
                    xtl[i] = ''

        if len(xticks) or len(xtick_labels):
            plt.xticks(xtx, xtl)

        plt.xticks(rotation=self.xticklabels_rotation)

        ytx, ytl = plt.yticks()

        if len(yticks):
            ytx = yticks
        if len(ytick_labels):
            ytl = ytick_labels

        if len(yticks) or len(ytick_labels):
            plt.yticks(ytx, ytl)

        plt.yticks(rotation=self.yticklabels_rotation)
