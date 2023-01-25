from tabplot import Plot

class BarPlot(Plot):
    bar_width: float

    def __init__(self, **kwargs) -> None:
        self.bar_width = 0.25
        super().__init__(**kwargs)


    def _plot_data(self, ax, xs, ys, labels, zorders):
        lines = []
        bar_count = 0
        for x,y,label in zip(xs,ys, labels):
            indices = list(range(1,len(y)+1))
            num_files = len(xs)
            width = self.bar_width
            shiftwidth = (num_files -1 )* width / 2.0 
            position = [ bar_count * width - shiftwidth + i for i in indices]
            line = ax.bar(position,y, width=width, label=label.replace('_', '-'))
            lines.append(line)
            bar_count = bar_count + 1
            # if args['xticks_column'] is not None: 
            #     plt.xticks(indices,xticks)
        return lines
