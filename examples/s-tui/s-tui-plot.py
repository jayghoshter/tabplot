#!/usr/bin/env python3

from tabplot import Plot, Figure
from matplotlib import pyplot as plt


fig = Figure(figsize=(4,6), legend_frameon = False)

pFan = (
    Plot(
        ylims=(0,5000),
        linestyles='dashed',
        ylabel = "Rotation Speed (RPM)",
    )
    .load(
        files=['./s-tui-log.txt'],
        columns=(-999, -1),
        labels=['Fan Speed (RPM)'],
        header=True,
    )
)

pTemp = ( 
    Plot(
        shape=(2,1),
        loc=(0,0),
        ylims=(0,100),
        colors='red',
        # legend_ncols=2,
        # legend_fontsize='x-small',
        twinx = pFan,
        ylabel = "Temperature (degC)"
    )
    .load(
        files=['./s-tui-log.txt'], 
        column_names=(None, 'temp:cpu'), 
        labels=['CPU Temperature (degC)'],
        header=True
    )
    .draw()
)

pPow = ( 
    Plot(
        shape=(2,1),
        loc=(1,0),
        ylims=(-10,65),
        ylabel = "Power (W)",
        twinx = pFan,
    )
    .load(
        files=['./s-tui-log.txt'], 
        columns=(-999, 'power:psys'), 
        labels=['PSYS Power Consumption (W)'],
        header=True
    )
    .draw()
)


(
    fig
    .load_subplots([pTemp, pPow])
    .legend(loc='lower right')
    .save('./s-tui2.pdf')
)
