from tplot.utils import make_iterable
from tplot.utils import readfile
from tplot.utils import normalize
from tplot.utils import scale_axis
from tplot.postprocessing import fit_lines
from tplot.postprocessing import extrapolate
from rich import print

from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from collections.abc import Iterable
from pathlib import Path

from typing import Optional, Tuple

class Plot:

    def __init__(self, **kwargs) -> None:

        # Labels
        self.title:str  = ''
        self.xlabel:str = ''
        self.ylabel:str = ''

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

        self.linestyles          :str|Iterable[str]     = []
        self.linewidths          :float|Iterable[float] = []
        self.markers             :str|Iterable[str]     = []
        self.markersize          :float|Iterable[float] = []
        self.marker_face_colors  :str|Iterable[str]     = []
        self.marker_edge_colors  :str|Iterable[str]     = []
        self.marker_edge_widths  :float|Iterable        = []
        self.line_color_indices  :int|Iterable[int]     = []
        self.line_color_indices_2:int|Iterable[int]     = []

        self.style:list[str] = ['science']
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

        # Filling and hatching
        self.fill:Optional[float|str] = None
        self.fill_color:Optional[str] = None
        self.fill_alpha:Optional[float] = 0.2
        self.hatch:Optional[str] = 'xxx'
        self.hatch_linewidth:Optional[float] = 0.5

        # Transforms
        # TODO: Convert these into methods
        self.smoothen_order:Optional[int] = None
        self.smoothen_npoints:int = 250
        self.normalize_y:Optional[bool|str|float] = None
        self.normalize_x:Optional[bool|str|float] = None
        self.xscale:Optional[float|np.ndarray|Path] = None
        self.yscale:Optional[float|np.ndarray|Path] = None
        self.extrapolate:Optional[str] = None
        self.xlogify:bool = False
        self.ylogify:bool = False

        # Addins
        # TODO: These too
        self.hlines:list[float] = []
        self.vlines:list[float] = []
        self.fit_lines:bool = False

        # TODO: Cleanup
        self.resample = False
        self.reaverage = False
        self.reaverage_cylindrical = False

        # File data parameters and dependents
        self.header            :bool           = False
        self.columns           :Tuple[int,int] = (0,1)
        self.xticks_column     :Optional[int]  = None
        self.xticklabels_column:Optional[int]  = None

        self.files    :list                  = []
        self.twinx    :list                  = []
        self._labels  :str|Iterable[str]     = []
        self._zorders :Iterable[float]       = []

        # Store ndarray data from all files (including twinx)
        self.file_data_list = []

        self.pointwise_annotation_padding: Optional[list[Tuple[float,float]]] = None
        self.pointwise_annotation_labels_column: Optional[int] = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def fromdict(self, data:dict):
        for key, value in data.items():
            setattr(self, key, value)
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

    def _update_params(self):

        n_total_files = len(self.files) + len(self.twinx)

        cmap = mpl.cm.get_cmap(name=self.colormap)
        if 'colors' in cmap.__dict__:
            # Discrete colormap
            self.COLORS = cmap.colors
        else:
            # Continuous colormap
            self.COLORS = [cmap(1.*i/(n_total_files-1)) for i in range(n_total_files)]

        if self.line_color_indices:
            self.line_color_indices = make_iterable(self.line_color_indices, 0, n_total_files, return_list = True)
            self.COLORS = [self.COLORS[i] for i in self.line_color_indices]

        # Ensure that our properties are of the right length. This was essential back when we used generators instead of property cyclers
        self.linestyles         = make_iterable(self.linestyles        , 'solid', n_total_files, return_list = True)
        self.linewidths         = make_iterable(self.linewidths        , 1      , n_total_files, return_list = True)
        self.markers            = make_iterable(self.markers           , None   , n_total_files, return_list = True)
        self.markersize         = make_iterable(self.markersize        , 6.0    , n_total_files, return_list = True)
        self.marker_edge_widths = make_iterable(self.marker_edge_widths, 1.0    , n_total_files, return_list = True)
        self.marker_face_colors = make_iterable(self.marker_face_colors, None   , n_total_files, return_list = True)
        self.marker_edge_colors = make_iterable(self.marker_edge_colors, None   , n_total_files, return_list = True)

        # Create a cycler
        self.final_cycler = self._get_props_cycler()

        if self.twinx: 
            self.final_cycler2 = self.final_cycler[len(self.files):].concat(self.final_cycler[:len(self.files)])

    def _get_props_cycler(self):
        main_c =  cycler(
            color           = list(self.COLORS[:len(self.files + self.twinx)]),
            linestyle       = list(self.linestyles),
            linewidth       = list(self.linewidths),
            marker          = list(self.markers),
            markersize      = list(self.markersize),
            markeredgewidth = list(self.marker_edge_widths),
        )

        if list(filter(None, self.marker_edge_colors)):
            main_c = main_c + cycler(markeredgecolor = list(self.marker_edge_colors) )

        if list(filter(None, self.marker_face_colors)):
            main_c = main_c + cycler(markerfacecolor = list(self.marker_face_colors) )

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
        ax.set_prop_cycle(self.final_cycler)

        ax.set(title = self.title)
        ax.set(xlabel = self.xlabel)
        ax.set(ylabel = self.ylabel)

        if self.xlog:
            ax.set(xscale='log')
        if self.ylog:
            ax.set(yscale='log')

        ax.set_aspect(self.aspect)
        # TODO: Expose
        ax.autoscale(tight=True)

        if self.xlims:
            ax.set_xlim(self.xlims)
        if self.ylims:
            ax.set_ylim(self.ylims)

        if self.twinx:
            self.ax2 = ax.twinx()
            ax2 = self.ax2
            ax2.set_prop_cycle(self.final_cycler2)
            ax2 = self.ax2
            ax2.set(ylabel=self.y2label)

            if self.y2log:
                ax2.set(yscale="log")

            if self.y2lims:
                ax2.set_ylim(self.y2lims)

    def _process_files(self, files):
        xs = []
        ys = []

        for filename in files:

            file_data = readfile(filename, header=self.header)
            self.file_data_list.append(file_data)

            if file_data.ndim == 1:
                x = np.array([])
                y = file_data.astype('float64') if self.columns[1] != -1 else np.array([])
            else:
                x = file_data[self.columns[0]].astype('float64') if self.columns[0] != -1 else np.array([])
                y = file_data[self.columns[1]].astype('float64') if self.columns[1] != -1 else np.array([])
                xticks = file_data[self.xticks_column].astype('float64') if self.xticks_column is not None else np.array([])
                xticklabels = file_data[self.xticklabels_column] if self.xticks_column is not None else np.array([])

                if len(xticks) or len(xticklabels):
                    self._setup_ticks(xticks = xticks, xtick_labels = xticklabels)

            if self.normalize_y:
                y = normalize(y, self.normalize_y)
            if self.normalize_x:
                x = normalize(x, self.normalize_x)

            if self.xscale:
                x = scale_axis(x, self.xscale)
            if self.yscale:
                y = scale_axis(y, self.yscale)

            # Unlike --xlog and --ylog, these actually scale the data
            if self.xlogify:
                x = np.log10(x)
            if self.ylogify:
                y = np.log10(y)

            if self.smoothen_order:
                xsmooth = np.linspace(min(x), max(x), self.smoothen_npoints)
                x_new = np.array(x)
                y_new = np.array(y)
                ordering = x_new.argsort()
                x_new = x_new[ordering]
                y_new = y_new[ordering]
                spl = make_interp_spline(x_new, y_new, k=self.smoothen_order)  # type: BSpline
                ysmooth = spl(xsmooth)
                x = xsmooth
                y = ysmooth

            xs.append(x)
            ys.append(y)

        return xs, ys

    def _plot_data(self, ax, xs, ys, labels, zorders):
        lines = []
        for x,y,label,zorder in zip(xs,ys,labels, zorders):
            line = ax.plot(x, y, label=label.replace('_', '-'), zorder=zorder )
            lines.extend(line)

            if isinstance(self.fill, float):
                ax.fill_between(x, y, self.fill, interpolate=True, hatch=self.hatch, alpha=self.fill_alpha)
                plt.rcParams['hatch.linewidth'] = self.hatch_linewidth
            elif isinstance(self.fill, str): 
                xfill, yfill = readfile(self.fill)
                if xfill != x:
                    raise NotImplementedError("Interpolation between curves for filling yet to be implemented!")
                ax.fill_between(x, y, yfill, interpolate=True, hatch=self.hatch, alpha=self.fill_alpha)

        return lines

    def _plot_legend(self, ax, all_lines, legend=None):
        if legend is None:
            legend = self.legend
        all_labels = [l.get_label() for l in all_lines]
        ax.legend(all_lines,
                  all_labels,
                  loc=legend[0],
                  bbox_to_anchor=(float(legend[1]),float(legend[2])),
                  shadow=True,
                  fontsize=self.legend_size,
                  ncol=self.legend_ncol)

    def generate(self,):
        # PREPROCESSING:
        self._update_params()

        plt.style.use(self.style)

        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        self._setup_axes()
        self._setup_ticks()

        labels_iter = iter(self.labels)
        zorders_iter = iter(self.zorders)

        # PROCESSING:
        print(f"Processing files: {self.files}")
        xs, ys, = self._process_files(self.files)
        lines = self._plot_data(self.ax, xs, ys, labels_iter, zorders_iter)

        lines2 = []
        xs2 = []
        ys2 = []
        if self.twinx:
            xs2, ys2 = self._process_files(self.twinx)
            lines2 = self._plot_data(self.ax2, xs2, ys2, labels_iter, zorders_iter)

        self.lines = lines + lines2

        if self.show_legend:
            self._plot_legend(self.ax, self.lines)

        # POSTPROCESSING:
        if self.fit_lines:
            fit_lines(self.ax, xs, ys, self.xlog, self.ylog)

            if self.twinx:
                fit_lines(self.ax2, xs2, ys2, self.xlog, self.ylog)

        if self.extrapolate:
            extrapolate(self.ax, xs, ys, self.extrapolate)

        if self.pointwise_annotation_labels_column is not None:
            padding_iter = iter(self.pointwise_annotation_padding)
            for x, y, file_data in zip(xs + xs2, ys + ys2, self.file_data_list):
                annots = iter(file_data[self.pointwise_annotation_labels_column])
                for xi, yi, annot, xypads in zip(x, y, annots, padding_iter):
                    xlims = self.ax.get_xlim()
                    x_pad = (xlims[1] - xlims[0]) * xypads[0]
                    ylims = self.ax.get_ylim()
                    y_pad = (ylims[1] - ylims[0]) * xypads[0]
                    self.ax.annotate(annot, (xi + x_pad, yi + y_pad))

        # hlines and vlines are plotted at the end so that xlims and ylims are
        # not later modified by further plotting
        xlim = self.ax.get_xlim()
        self.ax.hlines(self.hlines, xlim[0], xlim[1])

        ylim = self.ax.get_ylim()
        self.ax.vlines(self.vlines, ylim[0], ylim[1])

        if self.reverse_x:
            xlim = self.ax.get_xlim()
            self.ax.set_xlim((xlim[1], xlim[0]))

        if self.reverse_y:
            ylim = self.ax.get_ylim()
            self.ax.set_ylim((ylim[1], ylim[0]))

        return self

    def display(self,):
        # Default legend position of below the plot doesn't show on frontend
        self._plot_legend(self.ax, self.lines, ('upper left', 0, 1))
        plt.show()
        return self

    def save(self, filename):
        self.fig.savefig(filename, dpi=300)
        return self

    def __rich_repr__(self):
        yield self.files
        yield "files", self.files

    def __del__(self):
        plt.close(self.fig)
