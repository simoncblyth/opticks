#!/usr/bin/env python 
"""

https://docs.pyvista.org/examples/01-filter/glyphs.html

"""

import pyvista as pv
from pyvista import examples 

mesh = examples.download_carotid().threshold(145, scalars="scalars")
mask = mesh['scalars'] < 210
mesh['scalars'][mask] = 0  # null out smaller vectors

# Make a geometric object to use as the glyph
geom = pv.Arrow()  # This could be any dataset

# Perform the glyph
glyphs = mesh.glyph(orient="vectors", scale="scalars", factor=0.003, geom=geom)

# plot using the plotting class
pl = pv.Plotter()
pl.add_mesh(glyphs, show_scalar_bar=False, lighting=False, cmap='coolwarm')
pl.camera_position = [(146.53, 91.28, 21.70),
                      (125.00, 94.45, 19.81),
                      (-0.086, 0.007, 0.996)]  # view only part of the vector field
cpos = pl.show()

