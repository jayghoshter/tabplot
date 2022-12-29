# tabplot

A wrapper around matplotlib/numpy/scipy to make plotting csv-style tabular data easier. This project was initially just a `plot` python script in my `~/bin` folder. Since it kept growing, I decided to move it into its own separate package.

# Dependencies
```
matplotlib
numpy
scipy
SciencePlots
numexpr
```

# Notes/Issues
- Make sure to close the current plot with `.close()`. Unless garbage collected, plots may linger and calling `.show()` for some other object may also display a previous unclosed figure.
- hlines/vlines must be called after all plots/fit_lines etc are done so that no further changes to xlims, ylims are effected later.
- legends don't include hlines
