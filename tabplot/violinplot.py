from tabplot import Plot
import matplotlib.pyplot as plt
import numpy as np

class ViolinPlot(Plot):
    bar_width: float

    def __init__(self, **kwargs) -> None:
        self.bar_width = 0.25
        self.strip_xticklabels: bool = True
        super().__init__(**kwargs)
        self.show_legend = False

    def _plot_data(self, ax, xs, ys, labels, zorders, **kwargs):
        violin_parts = ax.violinplot([ y[~np.isnan(y)] for y in ys ], **kwargs)
        for pc, color in zip(violin_parts['bodies'], self.colors):
            pc.set_color(color)
            # pc.set_facecolor(color)
            # pc.set_edgecolor(color)
            # pc.set_linewidth()
            pc.set_alpha(0.5)
        return []

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
        indices = list(range(1, len(self.ys) + 1))

        if len(xticks):
            xtx = xticks
        else: 
            xtx = indices

        if len(xtick_labels):
            xtl = xtick_labels
        else: 
            xtl = [ f for f in self.labels ]

        plt.xticks(xtx, xtl, rotation=self.xticklabels_rotation)

        ytx, ytl = plt.yticks()

        if len(yticks):
            ytx = yticks
        if len(ytick_labels):
            ytl = ytick_labels

        if len(yticks) or len(ytick_labels):
            plt.yticks(ytx, ytl)

        plt.yticks(rotation=self.yticklabels_rotation)
