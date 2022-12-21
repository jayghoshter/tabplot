from tplot.utils import make_iterable
from tplot.utils import readfile
from tplot.utils import normalize
from tplot.utils import scale_axis
from tplot.utils import cmap_colors_to_hex

from tplot.postprocessing import fit_lines
from tplot.postprocessing import extrapolate

from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


class Plot:

    def __init__(self) -> None:

        # Labels
        self.title = ''
        self.xlabel = ''
        self.ylabel = ''

        # Size and dimension
        self.aspect = 'auto'
        self.figsize = [4.0, 3.0]
        self.xlims = None
        self.ylims = None

        # Direction
        self.reverse_x = False
        self.reverse_y = False

        # Ticks
        self.xticks = []
        self.yticks = []
        self.xtick_labels = []
        self.ytick_labels = []
        self.xlog = False
        self.ylog = False

        self.show_axis = True

        self.linestyles = []
        self.linewidths = []
        self.labels = []
        self.zorder = []
        self.markers = []
        self.markersize = []
        self.marker_face_colors = []
        self.marker_edge_colors = []
        self.marker_edge_widths = []
        self.line_color_indices = []
        self.line_color_indices_2 = []

        self.colormap = 'tab10'

        self.legend = ('upper center', '0.5', '-0.2')
        self.legend_frameon = False
        self.legend_size = 'medium'
        self.legend_ncol = 1

        # Transforms
        self.smoothen_order = None
        self.smoothen_npoints = 250
        self.normalize_y = None
        self.normalize_x = None
        self.xscale = None
        self.yscale = None
        self.extrapolate = None
        self.xlogify = False
        self.ylogify = False

        self.style = ['science', 'ieee']

        self.y2label = ''
        self.y2lims = None
        self.y2log = None
        self.colormap2 = 'tab10'

        # Addins
        self.hlines = []
        self.vlines = []
        self.fit_lines = False

        self.resample = False
        self.reaverage = False
        self.reaverage_cylindrical = False

        self.header = False
        self.files = []
        self.twinx = []

    def _update_params(self):

        n_total_files = len(self.files) + len(self.twinx)

        cmap = mpl.cm.get_cmap(name=self.colormap)
        if 'colors' in cmap.__dict__:
            # Discrete colormap
            self.COLORS = cmap.colors[:n_total_files]
        else:
            # Continuous colormap
            self.COLORS = [cmap(1.*i/(n_total_files-1)) for i in range(n_total_files)]

        if self.line_color_indices:
            self.line_color_indices = make_iterable(self.line_color_indices, 0, len(self.files), return_list = True)
            self.COLORS = [self.COLORS[i] for i in self.line_color_indices]

        self.color_cycler = cycler('color', self.COLORS)
        self.color_cycler2 = cycler('color', self.COLORS[len(self.files):] + self.COLORS[:len(self.files)])

        if not self.labels:
            self.labels = iter(self.files + self.twinx)
        else:
            self.labels = iter(self.labels)

        # Ensure that our properties are of the right length
        self.linestyles         = make_iterable(self.linestyles        , 'solid', n_total_files, return_list = True)
        self.linewidths         = make_iterable(self.linewidths        , 1      , n_total_files, return_list = True)
        self.markers            = make_iterable(self.markers           , None   , n_total_files, return_list = True)
        self.markersize         = make_iterable(self.markersize        , 6.0    , n_total_files, return_list = True)
        self.marker_edge_widths = make_iterable(self.marker_edge_widths, 1.0    , n_total_files, return_list = True)
        self.marker_face_colors = make_iterable(self.marker_face_colors, None   , n_total_files, return_list = True)
        self.marker_edge_colors = make_iterable(self.marker_edge_colors, None   , n_total_files, return_list = True)

        self.zorder             = self.zorder or iter(range(1,n_total_files + 1))

        # Create a cycler
        self.final_cycler = self._get_props_cycler()

        if self.twinx: 
            self.final_cycler2 = self.final_cycler[len(self.files):].concat(self.final_cycler[:len(self.files)])

    def _get_props_cycler(self):
        main_c =  cycler(
            color           = list(self.COLORS),
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

    def _setup_ticks(self,):
        # TODO: Move away from global plt
        xtx, xtl = plt.xticks()

        if self.xticks:
            xtx = self.xticks
        if self.xtick_labels:
            xtl = self.xtick_labels

        if self.xticks or self.xtick_labels:
            plt.xticks(xtx, xtl)

        ytx, ytl = plt.yticks()

        if self.yticks:
            ytx = self.yticks
        if self.ytick_labels:
            ytl = self.ytick_labels

        if self.yticks or self.ytick_labels:
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

        bar_count = 0

        for filename in files:
            print(f"Plotting: {filename}")

            # TODO: Handle xticks and xticklabels from files
            x, y, xticks, xticklabels = readfile(filename, columns=[0,1], header=self.header)

            if self.normalize_y:
                y = normalize(y, self.normalize_y)
            if self.normalize_x:
                x = normalize(x, self.normalize_x)

            # TODO: Allow this to be specified per plot line
            if self.xscale:
                if x == []:
                    x = [1] * len(y)
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

    def _plot_data(self, ax, xs, ys):
        lines = []
        for x,y,label in zip(xs,ys,self.labels):
            line = ax.plot(x, y, label=label.replace('_', '-'))
            lines.extend(line)
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

        # PROCESSING:
        xs, ys, = self._process_files(self.files)
        lines = self._plot_data(self.ax, xs, ys)

        lines2 = []
        xs2 = []
        ys2 = []
        if self.twinx:
            xs2, ys2 = self._process_files(self.twinx)
            lines2 = self._plot_data(self.ax2, xs2, ys2)

        self.lines = lines + lines2

        if self.legend:
            self._plot_legend(self.ax, self.lines)

        # POSTPROCESSING:
        if self.fit_lines:
            fit_lines(self.ax, xs, ys, self.xlog, self.ylog)

            if self.twinx:
                fit_lines(self.ax2, xs2, ys2, self.xlog, self.ylog)

        if self.extrapolate:
            extrapolate(self.ax, xs, ys, self.extrapolate)

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

    def display(self,):
        # Default legend position of below the plot doesn't show on frontend
        self._plot_legend(self.ax, self.lines, ('upper left', 0, 1))
        plt.show()

    def save(self, filename):
        self.fig.savefig(filename, dpi=300)

    def __rich_repr__(self):
        yield self.files
        yield "files", self.files
