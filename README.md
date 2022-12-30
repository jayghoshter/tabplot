# tabplot

**WORK IN PROGRESS**

When working from the terminal, I'd need to quickly (and beautifully) plot the data in csv-style text files. This meant good colors, axis labels, plot labels, adjustable axis limits, etc. I started with making a simple wrapper script around matplotlib to do this. Over time, I kept adding pre- and post-processing functionality such as scaling, normalizing, and linear regression, not to mention exposing more of matplotlib's properties and functions. Since the script kept growing, I decided to move it into its own separate package.

# Dependencies
```
matplotlib
numpy
scipy
scikit-learn
SciencePlots
numexpr
```

# Installation

You can install tabplot via testpypi

```
pip install matplotlib numpy scipy scikit-learn SciencePlots numexpr
pip install -i https://test.pypi.org/simple/ tabplot
```

# Examples

## Simple line plots

Assuming text data stored in csv-style files with the following assumptions:

- No header labels
- x,y data in columns 0,1 respectively
- Either space or comma separated

The following code will generate basic plots. For customization options please refer to the examples section or test cases.

```
from tabplot import Plot
plot = Plot()
plot.files = ['./file_1.csv', './file_2.csv'],
plot.linewidths = 2,
plot.labels = ['A', 'B']
plot.read()
plot.draw()
plot.show()
plot.close()
```

or

```
from tabplot import Plot
Plot(
    files = ['./file_1.csv', './file_2.csv'],
    linewidths = 2,
    labels = ['A', 'B']
).read().draw().show().close()
```

- Most used plot properties are exposed directly on the `Plot` object.
- `files`, `twinx`, and `labels` can be specified on `Plot` or in the `read()` method
- `read()` loads the data from the given files into `.xs` and `.ys`, which can be accessed and manipulated as wished.
- `setup()` creates a `.fig` and `.ax` if not already done
- `draw()` plots the lines etc. to the figure
- `save('file.pdf')` will write to `file.pdf`
- `show()` will display the plotted image

## Bar plots

```
from tabplot import BarPlot
files = ['dummy.csv']
BarPlot(bar_width = 0.5).read(files).draw().save('dummy.pdf').close()
```

- `BarPlot` is derived from `Plot`

# Notes/Issues
- Operator chaining is possible as seen above. 
- Currently only uses one subplot
- Make sure to close the current plot with `.close()`. Unless garbage collected, plots may linger and calling `.show()` for some other object may also display a previous unclosed figure.
- hlines/vlines must be called after all plots/fit_lines etc are done so that no further changes to xlims, ylims are effected later.
- legends don't include hlines
