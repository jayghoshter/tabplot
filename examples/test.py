from tplot import Plot

plot = Plot()
plot.files = ['./monodisperse']
plot.twinx = ['./polydisperse']
plot.linestyles = 'solid'
plot.linewidths = 2
plot.title = 'Title'
plot.xlabel = 'x axis'
plot.ylabel = 'y axis'
plot.fit_lines = True
plot.hlines = [ 3e-3, 4e-3 ]
plot.vlines = [ 2000, 4000 ]

plot.generate()
plot.display()
