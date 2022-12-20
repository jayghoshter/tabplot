from tplot import Plot

plot = Plot()
plot.files = ['./monodisperse']
plot.linewidths = 2
plot.title = 'Title'
plot.xlabel = 'x axis'
plot.ylabel = 'y axis'

plot.generate()
plot.display()
