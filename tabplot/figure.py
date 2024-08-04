from matplotlib import pyplot as plt
from tabplot import Plot
from typing import Optional, Tuple
import inspect

class Figure:

    figsize: Tuple[float, float] = (4.0, 3.0)
    style: Optional[list[str] | str] = None
    # # Allows overriding style by some settings given in this class such as font settings
    preload_style: bool = False

    savefig_bbox: str = "tight"

    font_family: str = "sans-serif"
    font_style: str = "normal"
    font_variant: str = "normal"
    font_weight: str = "normal"
    font_stretch: str = "normal"
    font_size: float = 10.0

    figure_dpi: int = 300

    legend_frameon: bool = True # if True, draw the legend on a background patch
    legend_framealpha: float = 0.8 # legend patch transparency
    legend_facecolor: str = "inherit" # inherit from axes.facecolor; or color spec
    legend_edgecolor: str = "inherit"
    legend_fancybox: bool = True # if True, use a rounded box for the
    legend_shadow: bool = False # if True, give background a shadow effect
    legend_numpoints: int = 1 # the number of marker points in the legend line
    legend_scatterpoints: int = 1 # number of scatter points
    legend_markerscale: float = 1.0 # the relative size of legend markers vs. original
    legend_fontsize: str = "medium"
    legend_labelcolor: Optional[str] = None
    legend_title_fontsize: Optional[str] = None # None sets to the same as the default axes.
    legend_borderpad: float = 0.4 # border whitespace
    legend_labelspacing: float = 0.5 # the vertical space between the legend entries
    legend_handlelength: float = 2.0 # the length of the legend lines
    legend_handleheight: float = 0.7 # the height of the legend handle
    legend_handletextpad: float = 0.8 # the space between the legend line and legend text
    legend_borderaxespad: float = 0.5 # the border between the axes and legend edge
    legend_columnspacing: float = 2.0 # column separation

    def __init__(self, **kwargs):
        default_instance_attributes = inspect.getmembers(self, lambda x: not(inspect.isroutine(x)))
        default_instance_attributes = {k:v for k,v in default_instance_attributes if not k.startswith('_')}
        self._default_instance_attributes = default_instance_attributes

        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            elif key in self._default_instance_attributes:
                setattr(self, key, value)
            else:
                raise NameError(f"No such attribute: {key}")

        self.fig = plt.figure(figsize=self.figsize)
        self.subplots = []

        if self.style and self.preload_style:
            plt.style.use(self.style)

        # TODO: Shrink code here
        # Set rc params
        self.setrc({
            "font.family": self.font_family,
            "font.style": self.font_style,
            "font.variant": self.font_variant,
            "font.weight": self.font_weight,
            "font.stretch": self.font_stretch,
            "font.size": self.font_size,
        })

        self.setrc({
            "legend.frameon": self.legend_frameon,
            "legend.framealpha": self.legend_framealpha,
            "legend.facecolor": self.legend_facecolor,
            "legend.edgecolor": self.legend_edgecolor,
            "legend.fancybox": self.legend_fancybox,
            "legend.shadow": self.legend_shadow,
            "legend.numpoints": self.legend_numpoints,
            "legend.scatterpoints": self.legend_scatterpoints,
            "legend.markerscale": self.legend_markerscale,
            "legend.fontsize": self.legend_fontsize,
            "legend.labelcolor": self.legend_labelcolor,
            "legend.title_fontsize": self.legend_title_fontsize,
            "legend.borderpad": self.legend_borderpad,
            "legend.labelspacing": self.legend_labelspacing,
            "legend.handlelength": self.legend_handlelength,
            "legend.handleheight": self.legend_handleheight,
            "legend.handletextpad": self.legend_handletextpad,
            "legend.borderaxespad": self.legend_borderaxespad,
            "legend.columnspacing": self.legend_columnspacing,
        })

        self.setrc({
                "figure.dpi": self.figure_dpi,
                "savefig.bbox": self.savefig_bbox
            })

        if self.style and not self.preload_style:
            plt.style.use(self.style)

    def setrc(self, rcdict):
        plt.rcParams.update(rcdict)
        return self

    def setrc_axes(self, rcdict):
        plt.rcParams.update({f"axes.{k}": v for k, v in rcdict.items()})
        return self

    def setrc_xtick(self, rcdict):
        plt.rcParams.update({f"xtick.{k}": v for k, v in rcdict.items()})
        return self

    def setrc_ytick(self, rcdict):
        plt.rcParams.update({f"ytick.{k}": v for k, v in rcdict.items()})
        return self

    def setrc_grid(self, rcdict):
        plt.rcParams.update({f"grid.{k}": v for k, v in rcdict.items()})
        return self

    def setrc_mathtext(self, rcdict):
        plt.rcParams.update({f"mathtext.{k}": v for k, v in rcdict.items()})
        return self

    def setrc_figure(self, rcdict):
        plt.rcParams.update({f"figure.{k}": v for k, v in rcdict.items()})
        return self

    def setrc_image(self, rcdict):
        plt.rcParams.update({f"image.{k}": v for k, v in rcdict.items()})
        return self

    def setrc_text(self, rcdict):
        plt.rcParams.update({f"text.{k}": v for k, v in rcdict.items()})
        return self

    def usetex(self, flag=True):
        plt.rcParams.update({"text.usetex": flag })
        return self

    def load_subplots(self, subplots:list[Plot], force_redraw=False):
        subplot_figs = set([ sub.ax.figure if sub.ax else None for sub in subplots ])

        if len(subplot_figs) > 1:
            raise ValueError("Subplots do not belong to the same figure!")
        elif len(subplot_figs) == 1:
            self.subplots = subplots
            prevFig = subplot_figs.pop()
            if prevFig is None:  # Not drawn
                for subplot in self.subplots:
                    subplot.fig = self.fig
            elif prevFig is not None: 
                if force_redraw:
                    plt.close(prevFig)
                    for subplot in self.subplots:
                        subplot.fig = self.fig
                        subplot.draw(renew_axis=True)
                else:
                    self.fig = prevFig
                    self.fig.set_size_inches(4,6) ## Doesn't work
        return self

    def draw(self, clean:bool=True, **kwargs):
        for subplot in self.subplots:
            subplot.draw(clean, **kwargs)
        return self

    def legend(self, **kwargs):
        for subplot in self.subplots:
            subplot.legend(**kwargs)
        return self

    def show(self,):
        self.fig.show()
        return self

    def save(self, fname, dpi=None):
        if dpi is None:
            dpi = self.figure_dpi
        self.fig.savefig(fname, dpi=dpi)
        return self
