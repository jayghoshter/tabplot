from tabplot.utils import make_iterable, fuzzy_str_to_idx, get_colors_from
from tabplot.utils import readfile, readheader
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

from typing import Optional, Tuple, Union, Literal, List
import inspect

class Plot:
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xlabel_loc: str = "center"
    ylabel_loc: str = "center"

    labels_from_headers: bool = True

    # Size and dimension
    aspect: str = "auto"
    shape: Tuple[int, int] = (1,1)
    loc: Tuple[int, int] = (0,0)
    rowspan:int = 1
    colspan:int = 1

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
    show_axis: bool = True

    colormap: str = "tab10"
    show_legend: bool = True
    combine_legends: bool = True

    legend_loc: str = "best"
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None
    legend_ncols: int = 1 

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
    _markevery: str | Iterable[str] = []

    _did_update_params: bool = False

    color_cycle_length:Optional[int] = None
    line_color_indices: int | Iterable[int] = []

    overwrite: bool = True


    fig = None
    twinx = None

    def __init__(self, **kwargs):
        self.ax = None

        ## NOTE: Need ys here so that inspect works as expected.
        self.xs: list[np.ndarray] = []
        self.ys: list[np.ndarray] = []
        self.lines = []
        self.aux_lines = []

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
    def labels(self):
        if self._labels:
            if len(self._labels) == self.n_total_lines():
                return self._labels
        # WARNING: Labels returns [] without filled p.ys
        # TODO: Fix this
        return [None] * self.n_total_lines()

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def zorders(self):
        if self._zorders:
            if len(self._zorders) == self.n_total_lines():
                return self._zorders
        return np.linspace(0, 1, self.n_total_lines())

    @zorders.setter
    def zorders(self, value):
        self._zorders = value

    def n_total_lines(self):
        return len(self.ys)

    @property
    def linestyles(self) -> Iterable:
        return make_iterable(
            self._linestyles, "solid", self.n_total_lines(), return_list=True
        )

    @linestyles.setter
    def linestyles(self, value):
        self._linestyles = value

    @property
    def markevery(self) -> Iterable:
        return make_iterable(
            self._markevery, 1, self.n_total_lines(), return_list=True
        )

    @markevery.setter
    def markevery(self, value):
        self._markevery = value

    @property
    def linewidths(self) -> Iterable:
        return make_iterable(
            self._linewidths, 1, self.n_total_lines(), return_list=True
        )

    @linewidths.setter
    def linewidths(self, value):
        self._linewidths = value

    @property
    def markers(self) -> Iterable:
        return make_iterable(
            self._markers, 'None', self.n_total_lines(), return_list=True
        )

    @markers.setter
    def markers(self, value):
        self._markers = value

    @property
    def markersizes(self) -> Iterable:
        return make_iterable(
            self._markersizes, 4.0, self.n_total_lines(), return_list=True
        )

    @markersizes.setter
    def markersizes(self, value):
        self._markersizes = value

    @property
    def markeredgewidths(self) -> Iterable:
        return make_iterable(
            self._markeredgewidths, 1.0, self.n_total_lines(), return_list=True
        )

    @markeredgewidths.setter
    def markeredgewidths(self, value):
        self._markeredgewidths = value

    @property
    def markeredgecolors(self) -> Iterable:
        if self._markeredgecolors:
            return make_iterable(
                self._markeredgecolors, None, self.n_total_lines(), return_list=True
            )
        else:
            return self.colors

    @markeredgecolors.setter
    def markeredgecolors(self, value):
        self._markeredgecolors = value

    @property
    def fillstyles(self) -> Iterable:
        return make_iterable(
            self._fillstyles, "full", self.n_total_lines(), return_list=True
        )

    @fillstyles.setter
    def fillstyles(self, value):
        self._fillstyles = value

    @property
    def markerfacecolors(self) -> Iterable:
        if self._markerfacecolors:
            return make_iterable(
                self._markerfacecolors, None, self.n_total_lines(), return_list=True
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
        #         for k, v in PlotAxis.__dict__.items()
        #         if isinstance(v, property)
        #     }
        # )

    def _update_params(self):

        # WARNING: in figures with multiple plots, the last plot setting overrides everything.
        # TODO: Fixme. Some of this might benefit moving to a new Figure class?

        if(self._did_update_params):
            return
        self._did_update_params = True

        color_cycle_length = self.color_cycle_length if self.color_cycle_length else self.n_total_lines()

        if not self.colors:
            self.colors = get_colors_from(self.colormap, color_cycle_length)

        if self.line_color_indices:
            self.line_color_indices = make_iterable(
                self.line_color_indices, 0, self.n_total_lines(), return_list=True
            )
            self.colors = [self.colors[i] for i in self.line_color_indices]

        ## Stripping it down to n_total_lines() 
        self.colors = make_iterable(self.colors, 'b', self.n_total_lines(), True)

        # Create a cycler
        self.props_cycler = self._get_props_cycler()


    def _get_props_cycler(self):

        main_c = cycler(
            color=self.colors[: self.n_total_lines()],
            linestyle=self.linestyles,
            linewidth=self.linewidths,
            marker=self.markers,
            markersize=self.markersizes,
            markeredgewidth=self.markeredgewidths,
            markeredgecolor=self.markeredgecolors[: self.n_total_lines()],
            markerfacecolor=self.markerfacecolors[: self.n_total_lines()],
            fillstyle=self.fillstyles,
            markevery=self.markevery,
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

        # WARNING: If setup is called more than once, this may be messed up
        if self.reverse_x:
            xlim = self.ax.get_xlim()
            self.ax.set_xlim((xlim[1], xlim[0]))

        if self.reverse_y:
            ylim = self.ax.get_ylim()
            self.ax.set_ylim((ylim[1], ylim[0]))

        if not self.show_axis:
            self.ax.axis('off')

        if self.sciticks:
            self.ax.ticklabel_format(style='sci', axis=self.sciticks, scilimits=(0,0))


    def load(
        self,
        files: Optional[list] = None,
        header: bool = False,
        columns: Tuple[int, int|list] | List[Tuple[int, int|list]] = (0, 1),
        column_names: Tuple[str, str|list[str]] | List[Tuple[str, str|list[str]]] | None = None,
        labels: Optional[list] = None,
        xticks_column: Optional[int] = None,
        xticklabels_column: Optional[int] = None,
        transpose: bool = False,
    ):

        if files is not None:
            self.files = files

        if labels is not None:
            self.labels = labels

        file_data_list = self._read_files(self.files, header)

        if header and column_names:
            columns = self._column_names_to_indices(column_names)

        if not labels:
            if header and self.labels_from_headers:
                self.labels = self._extract_header_labels(columns)

        if transpose:
            file_data_list = list(np.array(file_data_list).T)

        self.columns = columns # cache for fills
        self.xs, self.ys = self._extract_coordinate_data(file_data_list, columns)
        self._process_tick_data(file_data_list, xticks_column, xticklabels_column)

        return self

    def normalize_y(self, refValue=None):
        refValues = make_iterable(refValue, None, len(self.ys), return_list=True)
        self.ys = list(map(lambda y,ref: normalize(y, ref), self.ys, refValues))
        return self

    def normalize_y2(self, refValue=None):
        self.y2s = list(map(lambda y: normalize(y, refValue), self.y2s))
        return self

    def normalize_x(self, refValue=None):
        self.xs = list(map(lambda x: normalize(x, refValue), self.xs))
        return self

    def normalize_xy(self, refx=None, refy=None):
        self.normalize_x(refx)
        self.normalize_y(refy)
        return self

    def smoothen(self, order=3, npoints=250):
        self.xs, self.ys = smoothen_xys(self.xs, self.ys, order, npoints)

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

    def translate(
        self,
        dx: Union[float, np.ndarray] = 0.0,
        dy: Union[float, np.ndarray] = 0.0,
        ):
        """ Translate all xs and ys by dx and dy respectively """

        self.xs = [ x + dx for x in self.xs ]
        self.ys = [ y + dy for y in self.ys ]
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
        num = self.n_total_lines()
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

        self.headers=[]
        for filename in files:

            file_data = readfile(filename, header=header)
            file_data_list.append(file_data)

            if header and self.labels_from_headers:
                self.headers = readheader(filename)

        return file_data_list

    def _column_names_to_indices(
        self,
        column_names: Tuple[str|None, str|list[str]] | List[Tuple[str|None,str|list[str]]] = (0, 1),
    ):
        column_idx_sentinel = -999

        if isinstance(column_names, tuple):
            column_names_iter = [column_names] * len(self.headers)
        elif isinstance(column_names, list):
            assert len(column_names) == len(self.headers)
            column_names_iter = column_names


        indices = []
        for file_headers,col_names in zip(self.headers, column_names_iter):
            if isinstance(col_names[1], list):
                indices.append((fuzzy_str_to_idx(col_names[0], file_headers), [ fuzzy_str_to_idx(x, file_headers) for x in col_names[1]]))
            else:
                indices.append((fuzzy_str_to_idx(col_names[0], file_headers), fuzzy_str_to_idx(col_names[1], file_headers)))

        return indices

    def _extract_header_labels(
        self,
        columns: Tuple[int, int|list] | List[Tuple[int,int|list]] = (0, 1),
    ):

        column_idx_sentinel = -999

        if isinstance(columns, tuple):
            columns_iter = [columns] * len(self.headers)
        elif isinstance(columns, list):
            assert len(columns) == len(self.headers)
            columns_iter = columns

        headerlabels = []
        for file_headers,cols in zip(self.headers, columns_iter):
            if isinstance(cols[1], list):
                for ycol in cols[1]:
                    headerlabels.append(file_headers[ycol] if ycol != column_idx_sentinel else '---')
            else:
                headerlabels.append(file_headers[cols[1]] if cols[1] != column_idx_sentinel else '---')

        return headerlabels

    def _extract_coordinate_data(
        self,
        file_data_list,
        columns: Tuple[int, int|list] | List[Tuple[int,int|list]] = (0, 1),
    ):

        xs = []
        ys = []

        if isinstance(columns, tuple):
            columns_iter = [columns] * len(file_data_list)
        elif isinstance(columns, list):
            assert len(columns) == len(file_data_list)
            columns_iter = columns

        column_idx_sentinel = -999

        for file_data,cols in zip(file_data_list, columns_iter):
            if file_data.ndim == 1:
                x = np.array([])
                y = file_data.astype("float64") if cols[1] != column_idx_sentinel else np.array([])
                xs.append(x)
                ys.append(y)
            else:
                if isinstance(cols[1], list):
                    x = (
                        file_data[cols[0]].astype("float64")
                        if cols[0] != column_idx_sentinel
                        else np.array([])
                    )
                    for ycol in cols[1]:
                        y = (
                            file_data[ycol].astype("float64")
                            if ycol != column_idx_sentinel
                            else np.array([])
                        )
                        xs.append(x)
                        ys.append(y)
                else:
                    x = (
                        file_data[cols[0]].astype("float64")
                        if cols[0] != column_idx_sentinel
                        else np.array([])
                    )
                    y = (
                        file_data[cols[1]].astype("float64")
                        if cols[1] != column_idx_sentinel
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
        self.aux_lines = lines

        return self

    def extrapolate(self, kind="linear"):
        extrapolate(self.ax, self.xs, self.ys, kind)
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

    @property
    def destdir(self) -> Optional[Path]:
        return self._destdir

    @destdir.setter
    def destdir(self, value):
        if value is not None:
            self._destdir = Path(value)
        else:
            self._destdir = Path(".")

    def setup(self, clean: bool = True, renew_axis:bool = False):

        self._update_params()

        if not self.ax or renew_axis:
            self.ax = plt.subplot2grid(self.shape, self.loc, rowspan=self.rowspan, colspan=self.colspan, fig=self.fig)
        else:
            if clean:
                self.ax.cla()

        self._setup_axes()
        self._setup_ticks()

        return self

    def draw(self, clean: bool = True, renew_axis:bool = False, **kwargs):
        if not self.ys:
            return self

        self.setup(clean, renew_axis)

        labels_iter = iter(self.labels)
        zorders_iter = iter(self.zorders)

        lines = self._plot_data(self.ax, self.xs, self.ys, labels_iter, zorders_iter, **kwargs)
        self.lines = lines 

        if self.twinx:
            self.twinx.ax = self.ax.twinx()
            self.twinx.draw(clean, **kwargs) # NOTE: It needs to be previously loaded
            self.lines += self.twinx.lines

        return self

    def legend(self, **kwargs):
        self._plot_legend(self.ax, self.lines+self.aux_lines, **kwargs)
        return self

    def show(
        self,
    ):
        if not self.ys:
            return self

        if self.show_legend:
            if self.combine_legends:
                self._plot_legend(self.ax, self.lines+self.aux_lines, loc="best")
            else:
                self._plot_legend(self.ax, self.lines+self.aux_lines, loc="best")

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
                self._plot_legend(self.ax, self.lines + self.aux_lines)
            else:
                self._plot_legend(self.ax)

        if destdir is None:
            destdir = self.destdir
        else:
            destdir = Path(destdir)

        destdir.mkdir(exist_ok=True)

        if Path(destdir / filename).exists() and not overwrite:
            return self
        else:
            plt.savefig(
                destdir / filename, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches
            )
            print(f"Saved as {destdir / filename}\n")
            return self

    def __rich_repr__(self):
        yield self.files
        yield "files", self.files

    def __del__(self):
        plt.close()

    def close(self):
        plt.close()
