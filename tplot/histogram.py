from tplot import Plot
import numpy as np
from typing import Optional, Tuple

class Histogram(Plot):
    def __init__(self, **kwargs) -> None:
        self.bins:int = 20
        self.stacked:bool = False
        self.density:bool = True

        # TODO: 
        self.range:Optional[Tuple[float, float]] = None
        self.weights:Optional[np.ndarray] = None
        self.cumulative:bool = True
        self.bottom:Optional[np.ndarray|float] = True
        self.histtype:str = 'bar'
        self.align = 'mid'
        self.orientation = 'vertical'
        self.rwidth :Optional[float] = None
        self.log:bool = False

        super().__init__(**kwargs)

        self.columns = (-1,0)
        self.show_legend = False

    def _plot_data(self, ax, xs, ys, labels, zorders):
        lines = []
        n,bins,patches = ax.hist(ys, 
                                 bins=self.bins, 
                                 stacked=self.stacked, 
                                 density=self.density,
                                 label=list(labels))

        self._plot_legend(ax)

        return lines
