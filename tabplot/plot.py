from tabplot.utils import make_iterable
from tabplot.utils import readfile
from tabplot.utils import normalize
from tabplot.utils import scale_axis
from tabplot.utils import smoothen_xys
from tabplot.utils import strim
from tabplot.postprocessing import fit_lines
from tabplot.postprocessing import extrapolate
from rich import print

from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from collections.abc import Iterable
from pathlib import Path

from typing import Optional, Tuple, Union

class Plot:

    def __init__(self, **kwargs) -> None:

        print("Initializing Plot")

        # Labels
        self.title:str  = ''
        self.xlabel:str = ''
        self.ylabel:str = ''
        self.xlabel_loc:str = 'center'
        self.ylabel_loc:str = 'center'

        # Size and dimension
        self.aspect :str                          = 'auto'
        self.figsize:Tuple[float, float]          = (4.0, 3.0)
        self.xlims  :Optional[Tuple[float,float]] = None
        self.ylims  :Optional[Tuple[float,float]] = None

        # Direction
        self.reverse_x:bool = False
        self.reverse_y:bool = False

        # Ticks
        self.xticks      :np.ndarray|list[float] = np.array([])
        self.yticks      :np.ndarray|list[float] = np.array([])
        self.xtick_labels:np.ndarray|list[str] = np.array([])
        self.ytick_labels:np.ndarray|list[int] = np.array([])
        self.xlog        :bool = False
        self.ylog        :bool = False

        # TODO: 
        self.show_axis:bool = True

        self._linestyles           :str|Iterable[str]     = []
        self._linewidths           :float|Iterable[float] = []
        self._markers              :str|Iterable[str]     = []
        self._markersizes          :float|Iterable[float] = []
        self._markerfacecolors     :str|Iterable[str]     = []
        self._markeredgecolors     :str|Iterable[str]     = []
        self._markeredgewidths     :float|Iterable        = []

        self.line_color_indices  :int|Iterable[int]     = []
        self.line_color_indices_2:int|Iterable[int]     = []

        self.style:Optional[list[str]|str] = None
        self.colormap:str = 'tab10'

        self.show_legend   :bool                             = True
        self.legend        :Tuple[str, str|float, str|float] = ('upper center', '0.5', '-0.2')
        self.legend_frameon:bool                             = False
        self.legend_size   :str                              = 'medium'
        self.legend_ncol   :int                              = 1

        # Twinx attributes
        self.y2label:str = ''
        self.y2lims :Optional[Tuple[float,float]] = None
        self.y2log:bool = False
        self.colormap2:str = 'tab10'
        self.colors: list = []

        # Filling and hatching
        self.fill:Optional[float|str] = None
        self.fill_color:Optional[str] = None
        self.fill_alpha:Optional[float] = 0.2
        self.hatch:Optional[str] = 'xxx'
        self.hatch_linewidth:Optional[float] = 0.5
        self.hatch_color:Optional[str|tuple] = 'black'

        self.files    :list                  = []
        self.twinx    :list                  = []
        self._labels  :str|Iterable[str]     = []
        self._zorders :Iterable[float]       = []
        self._destdir = Path('.')

        # Store ndarray data from all files (including twinx)
        self.file_data_list = []

        self.xs :list[np.ndarray] = []
        self.ys :list[np.ndarray] = []
        self.x2s :list[np.ndarray] = []
        self.y2s :list[np.ndarray] = []

        self.fig:Optional[plt.Figure] = None
        self.ax  = None
        self.ax2 = None
        self.lines = []
        self.aux_lines = []

        for key, value in kwargs.items():
            if key in self.__dict__: 
                setattr(self, key, value)
            elif key in [p for p in dir(self.__class__) if isinstance(getattr(self.__class__,p),property)]:
                setattr(self, key, value)
            else: 
                raise NameError(f"No such attribute: {key}")

    @property
    def destdir(self):
        return self._destdir

    @destdir.setter
    def destdir(self, value):
        self._destdir = Path(value)

    def setup(self, clean:bool = True):

        self._update_params()

        if self.style:
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
            if len(self._labels) == len(self.files + self.twinx): 
                return self._labels
        return self.files + self.twinx

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def zorders(self):
        if self._zorders:
            if len(self._zorders) == len(self.files + self.twinx):
                return self._zorders
        return np.linspace(0,1,len(self.files + self.twinx))

    @zorders.setter
    def zorders(self, value):
        self._zorders= value

    @property
    def n_total_files(self):
        return len(self.files) + len(self.twinx)

    @property
    def linestyles(self):
        return make_iterable(self._linestyles, 'solid', self.n_total_files, return_list = True)

    @linestyles.setter
    def linestyles(self, value):
        self._linestyles = value 

    @property
    def linewidths(self):
        return make_iterable(self._linewidths, 1, self.n_total_files, return_list = True)

    @linewidths.setter
    def linewidths(self, value):
        self._linewidths = value 
        
    @property
    def markers(self):
        return make_iterable(self._markers, None, self.n_total_files, return_list = True)

    @markers.setter
    def markers(self, value):
        self._markers = value 

    @property
    def markersizes(self):
        return make_iterable(self._markersizes, 4.0, self.n_total_files, return_list = True)

    @markersizes.setter
    def markersizes(self, value):
        self._markersizes = value 

    @property
    def markeredgewidths(self):
        return make_iterable(self._markeredgewidths, 1.0, self.n_total_files, return_list = True)

    @markeredgewidths.setter
    def markeredgewidths(self, value):
        self._markeredgewidths = value 

    @property
    def markeredgecolors(self):
        return make_iterable(self._markeredgecolors, None, self.n_total_files, return_list = True)

    @markeredgecolors.setter
    def markeredgecolors(self, value):
        self._markeredgecolors = value 

    @property
    def markerfacecolors(self):
        return make_iterable(self._markerfacecolors, None, self.n_total_files, return_list = True)

    @markerfacecolors.setter
    def markerfacecolors(self, value):
        self._markerfacecolors = value 


    def _update_params(self):

        n_total_files = len(self.files) + len(self.twinx)

        cmap = mpl.cm.get_cmap(name=self.colormap)
        if 'colors' in cmap.__dict__:
            # Discrete colormap
            self.colors = cmap.colors
        else:
            # Continuous colormap
            self.colors = [cmap(1.*i/(n_total_files-1)) for i in range(n_total_files)]

        if self.line_color_indices:
            self.line_color_indices = make_iterable(self.line_color_indices, 0, n_total_files, return_list = True)
            self.colors = [self.colors[i] for i in self.line_color_indices]

        # Create a cycler
        self.final_cycler = self._get_props_cycler()

        if self.twinx: 
            self.final_cycler2 = self.final_cycler[len(self.files):].concat(self.final_cycler[:len(self.files)])

    def _get_props_cycler(self):
        main_c =  cycler(
            color           = list(self.colors[:len(self.files + self.twinx)]),
            linestyle       = self.linestyles,
            linewidth       = self.linewidths,
            marker          = self.markers,
            markersize      = self.markersizes,
            markeredgewidth = self.markeredgewidths,
        )

        if list(filter(None, self.markeredgecolors)):
            main_c = main_c + cycler(markeredgecolor = self.markeredgecolors)

        if list(filter(None, self.markerfacecolors)):
            main_c = main_c + cycler(markerfacecolor = self.markerfacecolors)

        return main_c

    def _setup_ticks(self, xticks=None, yticks=None, xtick_labels=None, ytick_labels=None):
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

        ytx, ytl = plt.yticks()

        if len(yticks):
            ytx = yticks
        if len(ytick_labels):
            ytl = ytick_labels

        if len(yticks) or len(ytick_labels):
            plt.yticks(ytx, ytl)

    def _setup_axes(self,):
        ax = self.ax
        self.ax.set_prop_cycle(self.final_cycler)

        ax.set(title = self.title)
        ax.set_xlabel(self.xlabel, loc=self.xlabel_loc)
        ax.set_ylabel(self.ylabel, loc=self.ylabel_loc)

        if self.xlog:
            ax.set(xscale='log')
        if self.ylog:
            ax.set(yscale='log')

        # TODO: Understand and expose
        ax.set_aspect(self.aspect)
        # TODO: Expose
        ax.autoscale(tight=True)

        if self.xlims:
            ax.set_xlim(self.xlims)
        if self.ylims:
            ax.set_ylim(self.ylims)

        if self.twinx:
            self.ax2 = ax.twinx()
            self.ax2.set_prop_cycle(self.final_cycler2)
            ax2 = self.ax2
            ax2 = self.ax2
            ax2.set(ylabel=self.y2label)

            if self.y2log:
                ax2.set(yscale="log")

            if self.y2lims:
                ax2.set_ylim(self.y2lims)

        if self.reverse_x:
            xlim = self.ax.get_xlim()
            self.ax.set_xlim((xlim[1], xlim[0]))

        if self.reverse_y:
            ylim = self.ax.get_ylim()
            self.ax.set_ylim((ylim[1], ylim[0]))

    def read(self, 
             files             :Optional[list] = None,
             twinx             :Optional[list] = None,
             header            :bool           = False,
             columns           :Tuple[int,int] = (0,1),
             labels            :Optional[list] = None,
             xticks_column     :Optional[int]  = None,
             xticklabels_column:Optional[int]  = None, ):

        if files is not None:
            self.files = files

        if twinx is not None:
            self.twinx = twinx

        if labels is not None:
            self.labels = labels

        file_data_list = self._read_files(self.files, header)
        self.xs, self.ys = self._extract_coordinate_data(file_data_list, columns)
        self._process_tick_data(file_data_list, xticks_column, xticklabels_column)

        file_data_list_2 = self._read_files(self.twinx, header)
        self.x2s, self.y2s = self._extract_coordinate_data(file_data_list_2, columns)
        self._process_tick_data(file_data_list_2, xticks_column, xticklabels_column)

        self.file_data_list = file_data_list + file_data_list_2

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
        self.x2s = list(map(lambda z: scale_axis(z, x), self.x2s))
        self.y2s = list(map(lambda z: scale_axis(z, y), self.y2s))
        return self

    def multiscale(self, x:Union[float,Iterable[float]]=1.0, y:Union[float,Iterable[float]]=1.0):
        num = len(self.files) + len(self.twinx)
        x   = make_iterable(x, 1.0, num, return_list = False)
        y   = make_iterable(y, 1.0, num, return_list = False)

        self.xs = list(map(scale_axis, self.xs, x))
        self.ys = list(map(scale_axis, self.ys, y))
        self.x2s = list(map(scale_axis, self.x2s, x))
        self.y2s = list(map(scale_axis, self.y2s, y))

        return self

    # TODO: implement xlogify

    def trim(self, condition:str):
        """
        Given a condition such as x<0.5, y>0.6 etc,
        "trims" the xs, and ys data to the given condition
        - Only one condition is applied to ALL the lines
        """
        self.xs, self.ys = strim(self.xs, self.ys, condition)
        return self

    def _read_files(self,
                    files             :list[Path],
                    header            :bool           = False,):

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


    def _extract_coordinate_data(self,
                                 file_data_list,
                                 columns           :Tuple[int,int] = (0,1),
                                 ):

        xs = []
        ys = []
        for file_data in file_data_list:
            if file_data.ndim == 1:
                x = np.array([])
                y = file_data.astype('float64') if columns[1] != -1 else np.array([])
            else:
                x = file_data[columns[0]].astype('float64') if columns[0] != -1 else np.array([])
                y = file_data[columns[1]].astype('float64') if columns[1] != -1 else np.array([])

            xs.append(x)
            ys.append(y)

        return xs, ys

    def _process_tick_data(self, 
                           file_data_list,
                           xticks_column     :Optional[int]  = None,
                           xticklabels_column:Optional[int]  = None, ):
        for file_data in file_data_list:
            # If we have more than just y-data
            if file_data.ndim > 1:
                xticks = file_data[xticks_column].astype('float64') if xticks_column is not None else np.array([])
                xticklabels = file_data[xticklabels_column] if xticks_column is not None else np.array([])

                if len(xticks) or len(xticklabels):
                    self._setup_ticks(xticks = xticks, xtick_labels = xticklabels)

    def _plot_data(self, ax, xs, ys, labels, zorders):
        lines = []
        for x,y,label,zorder in zip(xs,ys,labels, zorders):
            line = ax.plot(x, y, label=label.replace('_', '-'), zorder=zorder )
            lines.extend(line)

            if isinstance(self.fill, float) or isinstance(self.fill, int):
                ax.fill_between(x, y, self.fill, interpolate=True, hatch=self.hatch, alpha=self.fill_alpha)
                plt.rcParams['hatch.linewidth'] = self.hatch_linewidth
                plt.rcParams['hatch.color'] = self.hatch_color
            elif isinstance(self.fill, str): 
                xfill, yfill = readfile(self.fill)
                if xfill != x:
                    raise NotImplementedError("Interpolation between curves for filling yet to be implemented!")
                ax.fill_between(x, y, yfill, interpolate=True, hatch=self.hatch, alpha=self.fill_alpha)

        return lines

    def _plot_legend(self, ax, lines=None, legend=None):
        if legend is None:
            legend = self.legend
        if lines:
            all_labels = [l.get_label() for l in lines]
            ax.legend(lines,
                      all_labels,
                      loc=legend[0],
                      bbox_to_anchor=(float(legend[1]),float(legend[2])),
                      shadow=True,
                      fontsize=self.legend_size,
                      ncol=self.legend_ncol,
                      frameon=self.legend_frameon
                      )
        else:
            ax.legend(loc=legend[0],
                      bbox_to_anchor=(float(legend[1]),float(legend[2])),
                      shadow=True,
                      fontsize=self.legend_size,
                      ncol=self.legend_ncol,
                      frameon=self.legend_frameon
                      )


    def fit_lines(self, xlog=False, ylog=False, **kwargs): 
        lines = fit_lines(self.ax, self.xs, self.ys, xlog, ylog, **kwargs)

        lines2 = []
        if self.twinx:
            lines2 = fit_lines(self.ax2, self.x2s, self.y2s, xlog, ylog, **kwargs)

        self.aux_lines = lines + lines2

        return self

    def extrapolate(self, kind='linear'):
        extrapolate(self.ax, self.xs, self.ys, kind)
        if self.twinx: 
            extrapolate(self.ax2, self.x2s, self.y2s, kind)
        return self

    def annotate_pointwise(self, labels_column:int, paddings:list[Tuple[float,float]]):
        padding_iter = iter(paddings)
        file_data_iter = iter(self.file_data_list)
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
        self.ax.hlines(yvals, xlim[0], xlim[1], **kwargs)
        return self

    def vlines(self, xvals, **kwargs):
        ylim = self.ax.get_ylim()
        self.ax.vlines(xvals, ylim[0], ylim[1], **kwargs)
        return self

    def draw(self, clean:bool = True):
        # PREPROCESSING:

        self.setup(clean)

        labels_iter = iter(self.labels)
        zorders_iter = iter(self.zorders)

        # PROCESSING:
        print(f"Processing files: {self.files}")
        lines = self._plot_data(self.ax, self.xs, self.ys, labels_iter, zorders_iter)

        lines2 = []
        if self.twinx:
            lines2 = self._plot_data(self.ax2, self.x2s, self.y2s, labels_iter, zorders_iter)

        self.lines = lines + lines2

        return self

    def show(self,):
        if self.show_legend:
            self._plot_legend(self.ax, legend = ('upper left', 0, 1))

        plt.show()
        return self

    def save(self, filename, destdir=None, dpi=300, bbox_inches='tight', pad_inches=0.05):
        if self.show_legend:
            self._plot_legend(self.ax)

        if destdir is None:
            destdir = self.destdir
        else: 
            destdir = Path(destdir)

        destdir.mkdir(exist_ok=True)

        self.fig.savefig(destdir / filename,  dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        print(f"Saved as {destdir / filename}\n")
        return self

    def __rich_repr__(self):
        yield self.files
        yield "files", self.files

    def __del__(self):
        plt.close(self.fig)

    def close(self):
        plt.close(self.fig)
