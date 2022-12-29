import unittest

from pathlib import Path
from tabplot import Plot
import numpy as np
import shutil
import filecmp

class TestExamplePlot(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.resource_dir  = Path('./test_resources')
        self.output_dir    = Path('./test_output')
        self.reference_dir = Path('./tests/reference')

        self.resource_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.create_test_files()
        
    def create_test_files(self):
        x = np.linspace(0,10,100)
        y = (10*x) ** 2

        np.savetxt(self.resource_dir / 'test.csv',
                   np.stack([x,y],axis=1), 
                   delimiter = ',')

    def test_line_plot_basic(self):
        output_filename = 'line_plot_basic.png'
        reference_filename = 'line_plot_basic.png'
        plot = Plot(destdir = self.output_dir)
        plot.figsize = (4.0,4.0)
        plot.files = [ str(self.resource_dir / 'test.csv') ]
        plot.show_legend = False
        plot.read().draw().save(output_filename)

        comp = filecmp.cmp(self.reference_dir / reference_filename, self.output_dir / output_filename, shallow=False)

        self.assertTrue(comp)

    def test_files_spec_in_read(self):
        output_filename = 'line_plot_file_spec_in_read.png'
        reference_filename = 'line_plot_basic.png'
        plot = Plot(
            destdir = self.output_dir,
            figsize = (4.0,4.0),
            show_legend = False
        )
        plot.read(files = [ str(self.resource_dir / 'test.csv') ]).draw().save(output_filename)

        comp = filecmp.cmp(self.reference_dir / reference_filename, self.output_dir / output_filename, shallow=False)

        self.assertTrue(comp)
        

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(self.resource_dir)
        shutil.rmtree(self.output_dir)

