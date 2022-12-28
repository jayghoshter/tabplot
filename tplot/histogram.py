from tplot import Plot

class Histogram(Plot):
    def __init__(self, **kwargs) -> None:
        self.bins:int = 20
        super().__init__(**kwargs)

        self.columns = (-1,0)

    def _plot_data(self, ax, xs, ys, labels, zorders):
        lines = []
        for x,y,label in zip(xs,ys,labels):
            ax.hist(y, bins=self.bins, label=label)
        return lines
