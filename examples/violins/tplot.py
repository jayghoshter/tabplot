#!/usr/bin/env python

from tabplot.violinplot import ViolinPlot
from pathlib import Path

files = list(Path('.').glob('*.txt'))

p = ViolinPlot().read(files, labels=files).draw().save('vplot.pdf')


