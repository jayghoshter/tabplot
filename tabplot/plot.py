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
from collections.abc import Iterable
from pathlib import Path

from typing import Optional, Tuple, Union

class Plot:
    # Labels
    title:str  
    xlabel:str 
    ylabel:str 
    xlabel_loc:str 
    ylabel_loc:str 
    y2label_loc:str 

    # Size and dimension
    aspect :str                          
    figsize:Tuple[float, float]          
    xlims  :Optional[Tuple[float,float]] 
    ylims  :Optional[Tuple[float,float]] 

    # Direction
    reverse_x:bool 
    reverse_y:bool 

    # Ticks
    xticks      :np.ndarray|list[float] 
    yticks      :np.ndarray|list[float] 
    xtick_labels:np.ndarray|list[str] 
    ytick_labels:np.ndarray|list[int] 
    xlog        :bool 
    ylog        :bool 

    # TODO: 
    show_axis:bool 

    style:Optional[list[str]|str] 
    preload_style:bool 
    colormap:str 

    show_legend   :bool                             
    combine_legends :bool                             
    legend_loc : str
    legend_bbox_to_anchor: Optional[Tuple[float, float]]
    legend_ncols   :int                              
    legend_frameon: bool        
    legend_framealpha: float  
    legend_facecolor: str      
    legend_edgecolor: str      
    legend_fancybox: bool       
    legend_shadow : bool 
    legend_numpoints   : int   
    legend_scatterpoints  :int
    legend_markerscale   :float 
    legend_fontsize      :str 
    legend_labelcolor    :Optional[str]
    legend_title_fontsize : Optional[str]
    legend_borderpad:float
    legend_labelspacing   :float
    legend_handlelength   :float
    legend_handleheight   :float
    legend_handletextpad  :float
    legend_borderaxespad  :float
    legend_columnspacing  :float

    # Twinx attributes
    y2label:str 
    y2lims :Optional[Tuple[float,float]] 
    y2log:bool 
    colormap2:str 
    colors: list 

    # Filling and hatching
    fill:Optional[float|str] 
    fill_color:Optional[str] 
    fill_alpha:Optional[float] 
    hatch:Optional[str] 
    hatch_linewidth:Optional[float] 
    hatch_color:Optional[str|tuple] 

    files    :list                  
    twinx    :list                  

    _labels  :str|Iterable[str]     
    _zorders :Iterable[float]       
    _destdir : Optional[Path]

    _linestyles           :str|Iterable[str]     
    _linewidths           :float|Iterable[float] 
    _markers              :str|Iterable[str]     
    _markersizes          :float|Iterable[float] 
    _markerfacecolors     :str|Iterable[str]     
    _markeredgecolors     :str|Iterable[str]     
    _markeredgewidths     :float|Iterable[float]        
    _fillstyles           :str|Iterable[str]     

    line_color_indices  :int|Iterable[int]     
    line_color_indices_2:int|Iterable[int]     

    # Store ndarray data from all files (including twinx)
    # TODO: underscore?
    file_data_list:list 

    figure_dpi:int 

    # lines    : list 
    # aux_lines: list

    font_family  :str
    font_style   :str
    font_variant :str
    font_weight  :str
    font_stretch :str
    font_size    :float


    def __init__(self, **kwargs) -> None:

        print("Initializing Plot")

        # Labels
        self.title = ''
        self.xlabel = ''
        self.ylabel = ''
        self.xlabel_loc = 'center'
        self.ylabel_loc = 'center'
        self.y2label_loc = 'center'

        # Size and dimension
        self.aspect  = 'auto'
        self.figsize = (4.0, 3.0)
        self.xlims   = None
        self.ylims   = None

        # Direction
        self.reverse_x:bool = False
        self.reverse_y:bool = False

        # Ticks
        self.xticks       = np.array([])
        self.yticks       = np.array([])
        self.xtick_labels = np.array([])
        self.ytick_labels = np.array([])
        self.xlog         = False
        self.ylog         = False

        # TODO: 
        self.show_axis = True

        self._linestyles            = []
        self._linewidths            = []
        self._markers               = []
        self._markersizes           = []
        self._markerfacecolors      = []
        self._markeredgecolors      = []
        self._markeredgewidths      = []
        self._fillstyles            = []

        self.line_color_indices   = []
        self.line_color_indices_2 = []

        self.style = None
        self.preload_style = False         # Allows overriding style by some settings given in this class such as font settings
        self.colormap = 'tab10'

        self.show_legend    = True
        self.combine_legends  = True
        self.legend_loc            = 'best'
        self.legend_bbox_to_anchor = None
        self.legend_ncols    = 1
        self.legend_frameon        = True # if True, draw the legend on a background patch
        self.legend_framealpha     = 0.8 # legend patch transparency
        self.legend_facecolor      = 'inherit' # inherit from axes.facecolor; or color spec
        self.legend_edgecolor      = 'inherit'
        self.legend_fancybox       = True # if True, use a rounded box for the
        self.legend_shadow         = False # if True, give background a shadow effect
        self.legend_numpoints      = 1 # the number of marker points in the legend line
        self.legend_scatterpoints  = 1 # number of scatter points
        self.legend_markerscale    = 1.0 # the relative size of legend markers vs. original
        self.legend_fontsize       = 'medium'
        self.legend_labelcolor     = None
        self.legend_title_fontsize = None # None sets to the same as the default axes.
        self.legend_borderpad      = 0.4 # border whitespace
        self.legend_labelspacing   = 0.5 # the vertical space between the legend entries
        self.legend_handlelength   = 2.0 # the length of the legend lines
        self.legend_handleheight   = 0.7 # the height of the legend handle
        self.legend_handletextpad  = 0.8 # the space between the legend line and legend text
        self.legend_borderaxespad  = 0.5 # the border between the axes and legend edge
        self.legend_columnspacing  = 2.0 # column separation

        # Twinx attributes
        self.y2label = ''
        self.y2lims  = None
        self.y2log = False
        self.colormap2 = 'tab10'
        self.colors = []

        # Filling and hatching
        self.fill = None
        self.fill_color = None
        self.fill_alpha = 0.2
        self.hatch = 'xxx'
        self.hatch_linewidth = 0.5
        self.hatch_color = 'black'

        self.files     = []
        self.twinx     = []
        self._labels   = []
        self._zorders  = []
        self._destdir = Path('.')

        # Store ndarray data from all files (including twinx)
        # TODO: underscore?
        self.file_data_list = []

        self.xs :list[np.ndarray]    = []
        self.ys :list[np.ndarray]    = []
        self.x2s:list[np.ndarray]    = []
        self.y2s:list[np.ndarray]    = []

        self.figure_dpi = 300

        self.fig = None
        self.ax  = None
        self.ax2 = None
        self.lines = []
        self.aux_lines = []

        self.font_family  = 'sans-serif'
        self.font_style   = 'normal'
        self.font_variant = 'normal'
        self.font_weight  = 'normal'
        self.font_stretch = 'normal'
        self.font_size    = 10.0

        for key, value in kwargs.items():
            if key in self.__dict__: 
                setattr(self, key, value)
            elif key in [p for p in dir(self.__class__) if isinstance(getattr(self.__class__,p),property)]:
                setattr(self, key, value)
            else: 
                raise NameError(f"No such attribute: {key}")

    def from_dict(self, data):
        for key, value in data.items():
            if key in self.__dict__: 
                setattr(self, key, value)
            elif key in [p for p in dir(self.__class__) if isinstance(getattr(self.__class__,p),property)]:
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
            self._destdir = Path('.')

    def setup(self, clean:bool = True):

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

    def n_total_files(self):
        return len(self.files) + len(self.twinx)

    @property
    def linestyles(self) -> Iterable:
        return make_iterable(self._linestyles, 'solid', self.n_total_files(), return_list = True)

    @linestyles.setter
    def linestyles(self, value):
        self._linestyles = value 

    @property
    def linewidths(self) -> Iterable:
        return make_iterable(self._linewidths, 1, self.n_total_files(), return_list = True)

    @linewidths.setter
    def linewidths(self, value):
        self._linewidths = value 
        
    @property
    def markers(self) -> Iterable:
        return make_iterable(self._markers, None, self.n_total_files(), return_list = True)

    @markers.setter
    def markers(self, value):
        self._markers = value 

    @property
    def markersizes(self) -> Iterable:
        return make_iterable(self._markersizes, 4.0, self.n_total_files(), return_list = True)

    @markersizes.setter
    def markersizes(self, value):
        self._markersizes = value 

    @property
    def markeredgewidths(self) -> Iterable:
        return make_iterable(self._markeredgewidths, 1.0, self.n_total_files(), return_list = True)

    @markeredgewidths.setter
    def markeredgewidths(self, value):
        self._markeredgewidths = value 

    @property
    def markeredgecolors(self) -> Iterable:
        if self._markeredgecolors:
            return make_iterable(self._markeredgecolors, None, self.n_total_files(), return_list = True)
        else:
            return self.colors
            

    @markeredgecolors.setter
    def markeredgecolors(self, value):
        self._markeredgecolors = value 

    @property
    def fillstyles(self) -> Iterable:
        return make_iterable(self._fillstyles, 'full', self.n_total_files(), return_list = True) 

    @fillstyles.setter
    def fillstyles(self, value):
        self._fillstyles = value 

    @property
    def markerfacecolors(self) -> Iterable:
        if self._markerfacecolors:
            return make_iterable(self._markerfacecolors, None, self.n_total_files(), return_list = True)
        else:
            return self.colors

    @markerfacecolors.setter
    def markerfacecolors(self, value):
        self._markerfacecolors = value 

    # NOTE: While tempting, do not make this a property
    def get_properties(self):
        data = vars(self)
        data.update({
                        k: self.__getattribute__(k)
                        for k,v in Plot.__dict__.items()
                        if isinstance(v, property) 
                    })
        return data


    def setrc(self, rcdict):
        plt.rcParams.update(rcdict)          

    def setrc_axes(self, rcdict):
        plt.rcParams.update({ f'axes.{k}':v for k,v in rcdict.items()})          

    def setrc_xtick(self, rcdict):
        plt.rcParams.update({ f'xtick.{k}':v for k,v in rcdict.items()})          

    def setrc_ytick(self, rcdict):
        plt.rcParams.update({ f'ytick.{k}':v for k,v in rcdict.items()})          

    def setrc_grid(self, rcdict):
        plt.rcParams.update({ f'grid.{k}':v for k,v in rcdict.items()})          

    def setrc_mathtext(self, rcdict):
        plt.rcParams.update({ f'mathtext.{k}':v for k,v in rcdict.items()})          

    def setrc_figure(self, rcdict):
        plt.rcParams.update({ f'figure.{k}':v for k,v in rcdict.items()})          

    def setrc_image(self, rcdict):
        plt.rcParams.update({ f'image.{k}':v for k,v in rcdict.items()})          

    def setrc_text(self, rcdict):
        plt.rcParams.update({ f'text.{k}':v for k,v in rcdict.items()})          

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
        self.props_cycler = self._get_props_cycler()

        if self.twinx: 
            self.props_cycler2 = self.props_cycler[len(self.files):].concat(self.props_cycler[:len(self.files)])

        # Set rc params
        self.setrc(
            {
                'font.family'  : self.font_family,
                'font.style'   : self.font_style,
                'font.variant' : self.font_variant,
                'font.weight'  : self.font_weight,
                'font.stretch' : self.font_stretch,
                'font.size'    : self.font_size,
            }
        )

        self.setrc(
            {
                'legend.frameon' : self.legend_frameon,
                'legend.framealpha' : self.legend_framealpha,
                'legend.facecolor' : self.legend_facecolor,
                'legend.edgecolor' : self.legend_edgecolor,
                'legend.fancybox' : self.legend_fancybox,
                'legend.shadow' : self.legend_shadow,
                'legend.numpoints' : self.legend_numpoints,
                'legend.scatterpoints' : self.legend_scatterpoints,
                'legend.markerscale' : self.legend_markerscale,
                'legend.fontsize' : self.legend_fontsize,
                'legend.labelcolor' : self.legend_labelcolor,
                'legend.title_fontsize' : self.legend_title_fontsize,
                'legend.borderpad' : self.legend_borderpad,
                'legend.labelspacing' : self.legend_labelspacing,
                'legend.handlelength' : self.legend_handlelength,
                'legend.handleheight' : self.legend_handleheight,
                'legend.handletextpad' : self.legend_handletextpad,
                'legend.borderaxespad' : self.legend_borderaxespad,
                'legend.columnspacing' : self.legend_columnspacing,
            }
        )

        self.setrc(
            {
                'figure.dpi' : self.figure_dpi,
            }
        )


    def _get_props_cycler(self):
        main_c =  cycler(
            color           = self.colors[:len(self.files + self.twinx)],
            linestyle       = self.linestyles,
            linewidth       = self.linewidths,
            marker          = self.markers,
            markersize      = self.markersizes,
            markeredgewidth = self.markeredgewidths,
            markeredgecolor = self.markeredgecolors[:len(self.files + self.twinx)],
            markerfacecolor = self.markerfacecolors[:len(self.files + self.twinx)],
            fillstyle       = self.fillstyles
        )

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
        self.ax.set_prop_cycle(self.props_cycler)

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
            if not self.ax2:
                self.ax2 = ax.twinx()

            self.ax2.set_prop_cycle(self.props_cycler2)
            ax2 = self.ax2
            ax2.set_ylabel(self.y2label, loc=self.y2label_loc)

            if self.y2log:
                ax2.set(yscale="log")

            if self.y2lims:
                ax2.set_ylim(self.y2lims)

        # WARNING: If setup is called more than once, this may be messed up
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

    def _plot_legend(self, ax, lines=None, loc=None, bbox_to_anchor=None, ncols=None):
        if loc is None:
            loc = self.legend_loc

        if bbox_to_anchor is None:
            bbox_to_anchor = self.legend_bbox_to_anchor

        if ncols is None:
            ncols = self.legend_ncols


        if lines:
            all_labels = [l.get_label() for l in lines]
            ax.legend(
                lines,
                all_labels,
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                ncols = ncols
            )
        else:
            ax.legend(
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                ncols = ncols
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

        if not self.file_data_list: 
            return self

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
        if not self.file_data_list: 
            return self

        if self.show_legend:
            if self.combine_legends: 
                self._plot_legend(self.ax, self.lines + self.aux_lines, loc='best')
            else: 
                self._plot_legend(self.ax, loc='best')
                if self.twinx:
                    self._plot_legend(self.ax2, loc='best')

        plt.show()
        return self

    def save(self, filename, destdir=None, dpi=None, bbox_inches='tight', pad_inches=0.05):
        if not self.file_data_list: 
            return self

        if self.show_legend:
            if self.combine_legends: 
                self._plot_legend(self.ax, self.lines + self.aux_lines)
            else: 
                # TODO: Allow setting legend location for ax2 separately
                self._plot_legend(self.ax)
                if self.twinx:
                    self._plot_legend(self.ax2)

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
