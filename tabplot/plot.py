from tabplot.utils import make_iterable
from tabplot.utils import readfile
from tabplot.utils import normalize
from tabplot.utils import scale_axis
from tabplot.utils import smoothen_xys
from tabplot.utils import strim
from tabplot.postprocessing import fit_lines, parametric_line
from tabplot.postprocessing import extrapolate
from rich import print

from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np
from collections.abc import Iterable
from pathlib import Path

from typing import Optional, Tuple, Union, Literal
import inspect

class Plot:


    # Labels
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xlabel_loc: str = "center"
    ylabel_loc: str = "center"
    y2label_loc: str = "center"

    # Size and dimension
    aspect: str = "auto"
    figsize: Tuple[float, float] = (4.0, 3.0)
    xlims: Optional[Tuple[float, float]] = None
    ylims: Optional[Tuple[float, float]] = None

    # Direction
    reverse_x: bool = False
    reverse_y: bool = False

    # Ticks
    xticks: np.ndarray | list[float] = np.array([])
    yticks: np.ndarray | list[float] = np.array([])
    xtick_labels: np.ndarray | list[str] = np.array([])
    ytick_labels: np.ndarray | list[int] = np.array([])
    xticklabels_rotation: float = 0.0
    yticklabels_rotation: float = 0.0
    xlog: bool = False
    ylog: bool = False

    sciticks: Optional[Literal['x', 'y', 'both']] = None
    sciticks2: Optional[Literal['x', 'y', 'both']] = None

    show_axis: bool = True

    style: Optional[list[str] | str] = None

    # Allows overriding style by some settings given in this class such as font settings
    preload_style: bool = False
    colormap: str = "tab10"

    show_legend: bool = True
    combine_legends: bool = True
    legend_loc: str = "best"
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None
    legend_ncols: int = 1 
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

    # Twinx attributes
    y2label: str = ""
    y2lims: Optional[Tuple[float, float]] = None
    y2log: bool = False
    colormap2: str = "tab10"
    colors: list = []

    # Filling and hatching
    fill: Optional[float | str] = None
    fill_color: Optional[str] = None
    fill_alpha: Optional[float] = 0.2
    hatch: Optional[str] = "xxx"
    hatch_linewidth: Optional[float] = 0.5
    hatch_color: Optional[str | tuple] = "black"


    _labels: Iterable[str] = []
    _zorders: Iterable[float] = []
    _destdir: Optional[Path] = Path(".")

    _linestyles: str | Iterable[str] = []
    _linewidths: float | Iterable[float] = []
    _markers: str | Iterable[str] = []
    _markersizes: float | Iterable[float] = []
    _markerfacecolors: str | Iterable[str] = []
    _markeredgecolors: str | Iterable[str] = []
    _markeredgewidths: float | Iterable[float] = []
    _fillstyles: str | Iterable[str] = []

    color_cycle_length:Optional[int] = None
    line_color_indices: int | Iterable[int] = []
    line_color_indices_2: int | Iterable[int] = []

    # Store ndarray data from all files (including twinx)
    _file_data_list: list = []

    figure_dpi: int = 300

    # lines    : list
    # aux_lines: list

    font_family: str = "sans-serif"
    font_style: str = "normal"
    font_variant: str = "normal"
    font_weight: str = "normal"
    font_stretch: str = "normal"
    font_size: float = 10.0

    overwrite: bool = True

    def __init__(self, **kwargs) -> None:

        print("Initializing Plot")

        self.files: list = []
        self.twinx: list = []

        self.xs: list[np.ndarray] = []
        self.ys: list[np.ndarray] = []
        self.x2s: list[np.ndarray] = []
        self.y2s: list[np.ndarray] = []

        self.fig = None
        self.ax = None
        self.ax2 = None
        self.lines = []
        self.lines2 = []
        self.aux_lines = []
        self.aux_lines2 = []

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

    def from_dict(self, data):
        for key, value in data.items():
            if key in self.__dict__:
                setattr(self, key, value)
            elif key in self._default_instance_attributes:
                setattr(self, key, value)
            else:
                raise NameError(f"No such attribute: {key}")
        return self

    @property
    def destdir(self) -> Optional[Path]:
        return self._destdir

    @destdir.setter
    def destdir(self, value):
        if value is not None:
            self._destdir = Path(value)
        else:
            self._destdir = Path(".")

    def setup(self, clean: bool = True):

        if self.style and self.preload_style:
            plt.style.use(self.style)

        self._update_params()

        if self.style and not self.preload_style:
            plt.style.use(self.style)

        if not self.fig:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
        else:
            if clean:
                self.ax.cla()
                if self.ax2:
                    self.ax2.cla()

        self._setup_axes()
        self._setup_ticks()

        return self

    @property
    def labels(self):
        if self._labels:
            if len(self._labels) == self.n_total_files():
                return self._labels
        return [None] * self.n_total_files()

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def zorders(self):
        if self._zorders:
            if len(self._zorders) == self.n_total_files():
                return self._zorders
        # return np.linspace(0, 1, len(self.files + self.twinx))
        return np.linspace(0, 1, self.n_total_files())

    @zorders.setter
    def zorders(self, value):
        self._zorders = value

    def n_total_files(self):
        return len(self.ys) + len(self.y2s)

    @property
    def linestyles(self) -> Iterable:
        return make_iterable(
            self._linestyles, "solid", self.n_total_files(), return_list=True
        )

    @linestyles.setter
    def linestyles(self, value):
        self._linestyles = value

    @property
    def linewidths(self) -> Iterable:
        return make_iterable(
            self._linewidths, 1, self.n_total_files(), return_list=True
        )

    @linewidths.setter
    def linewidths(self, value):
        self._linewidths = value

    @property
    def markers(self) -> Iterable:
        return make_iterable(
            self._markers, None, self.n_total_files(), return_list=True
        )

    @markers.setter
    def markers(self, value):
        self._markers = value

    @property
    def markersizes(self) -> Iterable:
        return make_iterable(
            self._markersizes, 4.0, self.n_total_files(), return_list=True
        )

    @markersizes.setter
    def markersizes(self, value):
        self._markersizes = value

    @property
    def markeredgewidths(self) -> Iterable:
        return make_iterable(
            self._markeredgewidths, 1.0, self.n_total_files(), return_list=True
        )

    @markeredgewidths.setter
    def markeredgewidths(self, value):
        self._markeredgewidths = value

    @property
    def markeredgecolors(self) -> Iterable:
        if self._markeredgecolors:
            return make_iterable(
                self._markeredgecolors, None, self.n_total_files(), return_list=True
            )
        else:
            return self.colors

    @markeredgecolors.setter
    def markeredgecolors(self, value):
        self._markeredgecolors = value

    @property
    def fillstyles(self) -> Iterable:
        return make_iterable(
            self._fillstyles, "full", self.n_total_files(), return_list=True
        )

    @fillstyles.setter
    def fillstyles(self, value):
        self._fillstyles = value

    @property
    def markerfacecolors(self) -> Iterable:
        if self._markerfacecolors:
            return make_iterable(
                self._markerfacecolors, None, self.n_total_files(), return_list=True
            )
        else:
            return self.colors

    @markerfacecolors.setter
    def markerfacecolors(self, value):
        self._markerfacecolors = value

    # NOTE: While tempting, do not make this a property
    @classmethod
    def get_properties(cls):
        class_attributes = inspect.getmembers(cls, lambda x: not(inspect.isroutine(x) or isinstance(x, property)))
        class_attributes = {k:v for k,v in class_attributes if not k.startswith('_')}

        class_properties = inspect.getmembers(cls, lambda x: isinstance(x, property))
        class_properties = {k:getattr(cls, '_'+k) for k,v in class_properties if not k.startswith('_')}

        # NOTE: Python 3.9+ merging dicts
        return class_attributes | class_properties

        # data = vars(self)
        # data.update(
        #     {
        #         k: self.__getattribute__(k)
        #         for k, v in Plot.__dict__.items()
        #         if isinstance(v, property)
        #     }
        # )

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

    def _update_params(self):

        color_cycle_length = self.color_cycle_length if self.color_cycle_length else self.n_total_files()

        cmap = mpl.cm.get_cmap(name=self.colormap)
        if "colors" in cmap.__dict__:
            # Discrete colormap
            self.colors = cmap.colors
            self.colors = make_iterable(self.colors, 'b', color_cycle_length, True)
        else:
            # Continuous colormap
            self.colors = [
                cmap(1.0 * i / (color_cycle_length - 1)) for i in range(color_cycle_length)
            ]

        if self.line_color_indices:
            self.line_color_indices = make_iterable(
                self.line_color_indices, 0, self.n_total_files(), return_list=True
            )
            self.colors = [self.colors[i] for i in self.line_color_indices]

        # if self.color_cycle_length:
        #     self.colors = make_iterable(self.colors, 'b', self.color_cycle_length, True)
        self.colors = make_iterable(self.colors, 'b', self.n_total_files(), True)

        # Create a cycler
        self.props_cycler = self._get_props_cycler()

        if self.twinx:
            self.props_cycler2 = self.props_cycler[len(self.ys):].concat(
                self.props_cycler[:len(self.ys)]
            )

        # Set rc params
        self.setrc(
            {
                "font.family": self.font_family,
                "font.style": self.font_style,
                "font.variant": self.font_variant,
                "font.weight": self.font_weight,
                "font.stretch": self.font_stretch,
                "font.size": self.font_size,
            }
        )

        self.setrc(
            {
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
            }
        )

        self.setrc(
            {
                "figure.dpi": self.figure_dpi,
            }
        )

    def _get_props_cycler(self):
        main_c = cycler(
            color=self.colors[: self.n_total_files()],
            linestyle=self.linestyles,
            linewidth=self.linewidths,
            marker=self.markers,
            markersize=self.markersizes,
            markeredgewidth=self.markeredgewidths,
            markeredgecolor=self.markeredgecolors[: self.n_total_files()],
            markerfacecolor=self.markerfacecolors[: self.n_total_files()],
            fillstyle=self.fillstyles,
        )

        return main_c

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

        if len(xticks):
            xtx = xticks
        if len(xtick_labels):
            xtl = xtick_labels

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

    def _setup_axes(
        self,
    ):
        ax = self.ax
        self.ax.set_prop_cycle(self.props_cycler)

        ax.set(title=self.title)
        ax.set_xlabel(self.xlabel, loc=self.xlabel_loc)
        ax.set_ylabel(self.ylabel, loc=self.ylabel_loc)

        if self.xlog:
            ax.set(xscale="log")
        if self.ylog:
            ax.set(yscale="log")

        # TODO: Understand and expose
        ax.set_aspect(self.aspect)
        # TODO: Expose
        ax.autoscale(tight=True)

        if self.xlims:
            ax.set_xlim(self.xlims)
        if self.ylims:
            ax.set_ylim(self.ylims)

        if self.twinx:
            if not self.ax2:
                self.ax2 = ax.twinx()

            self.ax2.set_prop_cycle(self.props_cycler2)
            ax2 = self.ax2
            ax2.set_ylabel(self.y2label, loc=self.y2label_loc)

            if self.y2log:
                ax2.set(yscale="log")

            if self.y2lims:
                ax2.set_ylim(self.y2lims)

            ax2.autoscale(tight=True)

        # WARNING: If setup is called more than once, this may be messed up
        if self.reverse_x:
            xlim = self.ax.get_xlim()
            self.ax.set_xlim((xlim[1], xlim[0]))

        if self.reverse_y:
            ylim = self.ax.get_ylim()
            self.ax.set_ylim((ylim[1], ylim[0]))

        if not self.show_axis:
            self.ax.axis('off')
            if self.ax2:
                self.ax2.axis('off')

        if self.sciticks:
            self.ax.ticklabel_format(style='sci', axis=self.sciticks, scilimits=(0,0))

        if self.sciticks2: 
            if self.ax2: 
                self.ax2.ticklabel_format(style='sci', axis=self.sciticks, scilimits=(0,0))

    def read(
        self,
        files: Optional[list] = None,
        twinx: Optional[list] = None,
        header: bool = False,
        columns: Tuple[int, int] = (0, 1),
        labels: Optional[list] = None,
        xticks_column: Optional[int] = None,
        xticklabels_column: Optional[int] = None,
        transpose: bool = False,
    ):

        if files is not None:
            self.files = files

        if twinx is not None:
            self.twinx = twinx

        if labels is not None:
            self.labels = labels

        file_data_list = self._read_files(self.files, header)

        if transpose:
            file_data_list = list(np.array(file_data_list).T)

        self.xs, self.ys = self._extract_coordinate_data(file_data_list, columns)
        self._process_tick_data(file_data_list, xticks_column, xticklabels_column)

        file_data_list_2 = self._read_files(self.twinx, header)

        if transpose:
            file_data_list_2 = list(np.array(file_data_list_2).T)

        self.x2s, self.y2s = self._extract_coordinate_data(file_data_list_2, columns)
        self._process_tick_data(file_data_list_2, xticks_column, xticklabels_column)

        self._file_data_list = file_data_list + file_data_list_2

        return self

    def normalize_y(self, refValue=None):
        self.ys = list(map(lambda y: normalize(y, refValue), self.ys))
        self.y2s = list(map(lambda y: normalize(y, refValue), self.y2s))
        return self

    def normalize_x(self, refValue=None):
        self.xs = list(map(lambda x: normalize(x, refValue), self.xs))
        self.x2s = list(map(lambda x: normalize(x, refValue), self.x2s))
        return self

    def normalize_xy(self, refx=None, refy=None):
        self.normalize_x(refx)
        self.normalize_y(refy)
        return self

    def smoothen(self, order=3, npoints=250):
        self.xs, self.ys = smoothen_xys(self.xs, self.ys, order, npoints)
        self.x2s, self.y2s = smoothen_xys(self.x2s, self.y2s, order, npoints)
        return self

    def scale(self, x=1.0, y=1.0):
        self.xs = list(map(lambda z: scale_axis(z, x), self.xs))
        self.ys = list(map(lambda z: scale_axis(z, y), self.ys))
        return self

    def scale_twin(self, x=1.0, y=1.0):
        self.x2s = list(map(lambda z: scale_axis(z, x), self.x2s))
        self.y2s = list(map(lambda z: scale_axis(z, y), self.y2s))
        return self

    def scale_all(self, x=1.0, y=1.0):
        self.xs = list(map(lambda z: scale_axis(z, x), self.xs))
        self.ys = list(map(lambda z: scale_axis(z, y), self.ys))
        self.x2s = list(map(lambda z: scale_axis(z, x), self.x2s))
        self.y2s = list(map(lambda z: scale_axis(z, y), self.y2s))
        return self

    def multiscale(
        self,
        x: Union[float, Iterable[float]] = 1.0,
        y: Union[float, Iterable[float]] = 1.0,
    ):
        num = len(self.ys)
        x = make_iterable(x, 1.0, num, return_list=False)
        y = make_iterable(y, 1.0, num, return_list=False)

        self.xs = list(map(scale_axis, self.xs, x))
        self.ys = list(map(scale_axis, self.ys, y))

        return self

    def multiscale_all(
        self,
        x: Union[float, Iterable[float]] = 1.0,
        y: Union[float, Iterable[float]] = 1.0,
    ):
        num = self.n_total_files()
        x = make_iterable(x, 1.0, num, return_list=False)
        y = make_iterable(y, 1.0, num, return_list=False)

        self.xs = list(map(scale_axis, self.xs, x))
        self.ys = list(map(scale_axis, self.ys, y))
        self.x2s = list(map(scale_axis, self.x2s, x))
        self.y2s = list(map(scale_axis, self.y2s, y))

        return self

    def multiscale_twin(
        self,
        x: Union[float, Iterable[float]] = 1.0,
        y: Union[float, Iterable[float]] = 1.0,
    ):
        num = len(self.twinx)
        x = make_iterable(x, 1.0, num, return_list=False)
        y = make_iterable(y, 1.0, num, return_list=False)

        self.x2s = list(map(scale_axis, self.x2s, x))
        self.y2s = list(map(scale_axis, self.y2s, y))

        return self

    # TODO: implement xlogify

    def trim(self, condition: str):
        """
        Given a condition such as x<0.5, y>0.6 etc,
        "trims" the xs, and ys data to the given condition
        - Only one condition is applied to ALL the lines
        """
        self.xs, self.ys = strim(self.xs, self.ys, condition)
        return self

    def _read_files(
        self,
        files: list[Path],
        header: bool = False,
    ):

        """
        - Read a list of files
        - Save the data array
        - Return xs, ys, xticks, and xticklabels
        """

        file_data_list = []

        for filename in files:

            file_data = readfile(filename, header=header)
            file_data_list.append(file_data)

        return file_data_list

    def _extract_coordinate_data(
        self,
        file_data_list,
        columns: Tuple[int, int] = (0, 1),
    ):

        xs = []
        ys = []
        for file_data in file_data_list:
            if file_data.ndim == 1:
                x = np.array([])
                y = file_data.astype("float64") if columns[1] != -1 else np.array([])
            else:
                x = (
                    file_data[columns[0]].astype("float64")
                    if columns[0] != -1
                    else np.array([])
                )
                y = (
                    file_data[columns[1]].astype("float64")
                    if columns[1] != -1
                    else np.array([])
                )

            xs.append(x)
            ys.append(y)

        return xs, ys

    def _process_tick_data(
        self,
        file_data_list,
        xticks_column: Optional[int] = None,
        xticklabels_column: Optional[int] = None,
    ):
        if self.xticks or self.yticks:
            return

        for file_data in file_data_list:
            # If we have more than just y-data
            if file_data.ndim > 1:
                self.xticks = (
                    file_data[xticks_column].astype("float64")
                    if xticks_column is not None
                    else np.array([])
                )
                self.xtick_labels = (
                    file_data[xticklabels_column]
                    if xticklabels_column is not None
                    else np.array([])
                )

    def _plot_data(self, ax, xs, ys, labels, zorders):
        lines = []
        for x, y, label, zorder in zip(xs, ys, labels, zorders):
            if len(x) != 0:
                line = ax.plot(x, y, label=label, zorder=zorder)
            else: 
                line = ax.plot(y, label=label, zorder=zorder)
            lines.extend(line)

            if isinstance(self.fill, float) or isinstance(self.fill, int):
                ax.fill_between(
                    x,
                    y,
                    self.fill,
                    interpolate=True,
                    hatch=self.hatch,
                    alpha=self.fill_alpha,
                )
                plt.rcParams["hatch.linewidth"] = self.hatch_linewidth
                plt.rcParams["hatch.color"] = self.hatch_color
            elif isinstance(self.fill, str):
                xfill, yfill = readfile(self.fill)
                if xfill != x:
                    raise NotImplementedError(
                        "Interpolation between curves for filling yet to be implemented!"
                    )
                ax.fill_between(
                    x,
                    y,
                    yfill,
                    interpolate=True,
                    hatch=self.hatch,
                    alpha=self.fill_alpha,
                )

        return lines

    def _plot_legend(self, ax, lines=None, loc=None, bbox_to_anchor=None, ncols=None):
        if loc is None:
            loc = self.legend_loc

        if bbox_to_anchor is None:
            bbox_to_anchor = self.legend_bbox_to_anchor

        if ncols is None:
            ncols = self.legend_ncols

        if lines:
            all_labels = [l.get_label() for l in lines]
            if all([a.startswith('_') for a in all_labels]): 
                # Unlabeled
                # ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncols=ncols)
                pass
            else: 
                ax.legend(
                    lines, all_labels, loc=loc, bbox_to_anchor=bbox_to_anchor, ncols=ncols
                )
        else:
            ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncols=ncols)

    def fit_lines(self, xlog=False, ylog=False, **kwargs):
        lines = fit_lines(self.ax, self.xs, self.ys, xlog, ylog, **kwargs)

        lines2 = []
        if self.twinx:
            lines2 = fit_lines(self.ax2, self.x2s, self.y2s, xlog, ylog, **kwargs)

        self.aux_lines = lines + lines2

        return self

    def extrapolate(self, kind="linear"):
        extrapolate(self.ax, self.xs, self.ys, kind)
        if self.twinx:
            extrapolate(self.ax2, self.x2s, self.y2s, kind)
        return self

    def annotate_pointwise(
        self, labels_column: int, paddings: list[Tuple[float, float]]
    ):
        padding_iter = iter(paddings)
        file_data_iter = iter(self._file_data_list)
        for x, y, file_data in zip(self.xs, self.ys, file_data_iter):
            annots = iter(file_data[labels_column])
            for xi, yi, annot, xypads in zip(x, y, annots, padding_iter):
                xlims = self.ax.get_xlim()
                x_pad = (xlims[1] - xlims[0]) * xypads[0]
                ylims = self.ax.get_ylim()
                y_pad = (ylims[1] - ylims[0]) * xypads[0]
                self.ax.annotate(annot, (xi + x_pad, yi + y_pad))

        for x, y, file_data in zip(self.x2s, self.y2s, file_data_iter):
            annots = iter(file_data[labels_column])
            for xi, yi, annot, xypads in zip(x, y, annots, padding_iter):
                xlims = self.ax.get_xlim()
                x_pad = (xlims[1] - xlims[0]) * xypads[0]
                ylims = self.ax.get_ylim()
                y_pad = (ylims[1] - ylims[0]) * xypads[0]
                self.ax2.annotate(annot, (xi + x_pad, yi + y_pad))

        return self

    def hlines(self, yvals, **kwargs):
        xlim = self.ax.get_xlim()
        lineCollection = self.ax.hlines(yvals, xlim[0], xlim[1], **kwargs)
        # for line,label in zip(lineCollection.get_segments(), kwargs.get('labels')):
        #     line.set_label(label)
        # self.aux_lines.extend(lineCollection.get_segments())
        return self

    def vlines(self, xvals, **kwargs):
        ylim = self.ax.get_ylim()
        lineCollection = self.ax.vlines(xvals, ylim[0], ylim[1], **kwargs)
        # self.aux_lines.extend(lineCollection.get_segments())
        return self

    def mcline(self, m, c, label='mcline', zorder=0):
        line = parametric_line(
            self.ax, m, c, label=label, zorder=zorder
        )
        self.lines.extend(line)
        return self

    def draw(self, clean: bool = True, **kwargs):
        # PREPROCESSING:

        if not self.ys: 
            return self

        self.setup(clean)

        labels_iter = iter(self.labels)
        zorders_iter = iter(self.zorders)

        # PROCESSING:
        print(f"Processing files: {self.files}")
        lines = self._plot_data(self.ax, self.xs, self.ys, labels_iter, zorders_iter, **kwargs)
        self.lines = lines 

        lines2 = []
        if self.twinx:
            print(f"Processing twin files: {self.twinx}")
            lines2 = self._plot_data(
                self.ax2, self.x2s, self.y2s, labels_iter, zorders_iter, **kwargs
            )
            self.lines2 = lines2 

        return self

    def show(
        self,
    ):
        if not self.ys:
            return self

        if self.show_legend:
            if self.combine_legends:
                self._plot_legend(self.ax, self.lines + self.lines2 + self.aux_lines+self.aux_lines2, loc="best")
            else:
                self._plot_legend(self.ax, self.lines+self.aux_lines, loc="best")
                if self.twinx:
                    self._plot_legend(self.ax2, self.lines2+self.aux_lines2, loc="best")

        plt.show()
        return self

    def save(
        self, filename, destdir=None, dpi=None, bbox_inches="tight", pad_inches=0.05, overwrite=None
    ):
        if not self.ys:
            return self

        if overwrite is None:
            overwrite = self.overwrite

        # TODO: auto detect vs manual?
        # TODO: Combine vs not
        if self.show_legend:
            if self.combine_legends:
                # self._plot_legend(self.ax, self.lines + self.aux_lines)
                self._plot_legend(self.ax, self.lines + self.lines2 + self.aux_lines+self.aux_lines2)
            else:
                # TODO: Allow setting legend location for ax2 separately
                self._plot_legend(self.ax)
                # self._plot_legend(self.ax, self.lines+self.aux_lines, loc="best")
                if self.twinx:
                    self._plot_legend(self.ax2)
                    # self._plot_legend(self.ax2, self.lines2+self.aux_lines2, loc="best")

        if destdir is None:
            destdir = self.destdir
        else:
            destdir = Path(destdir)

        destdir.mkdir(exist_ok=True)

        if Path(destdir / filename).exists() and not overwrite:
            return self
        else:
            self.fig.savefig(
                destdir / filename, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches
            )
            print(f"Saved as {destdir / filename}\n")
            return self

    def __rich_repr__(self):
        yield self.files
        yield "files", self.files

    def __del__(self):
        plt.close(self.fig)

    def close(self):
        plt.close(self.fig)
