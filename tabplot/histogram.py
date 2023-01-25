from tabplot import Plot
import numpy as np
from typing import Optional, Tuple

class Histogram(Plot):
    bins:int
    stacked:bool
    density:bool

    def __init__(self, **kwargs) -> None:
        self.bins = 20
        self.stacked = False
        self.density = False

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

    def _plot_data(self, ax, xs, ys, labels, zorders):
        lines = []
        n,bins,patches = ax.hist(ys, 
                                 bins=self.bins, 
                                 stacked=self.stacked, 
                                 density=self.density,
                                 label=list(labels))

        return lines
