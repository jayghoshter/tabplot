from tplot import Plot

class Histogram(Plot):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.bins:int = 20

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.columns = (-1,0)

    def _plot_data(self, ax, xs, ys, labels, zorders):
        lines = []
        for x,y,label in zip(xs,ys,labels):
            ax.hist(y, bins=self.bins, label=label)
        return lines
