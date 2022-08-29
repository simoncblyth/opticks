#!/usr/bin/env python
"""
https://matplotlib.org/stable/gallery/shapes_and_collections/donut.html#sphx-glr-gallery-shapes-and-collections-donut-py


https://stackoverflow.com/questions/44025403/how-to-use-matplotlib-path-to-draw-polygon

"""
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def wise(v):
    return "CCW" if v == 1 else "CW"

def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))

Path = mpath.Path

fig, ax = plt.subplots()

inside_vertices = make_circle(0.5)
outside_vertices = make_circle(1.0)

codes = np.ones(len(inside_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
codes[0] = mpath.Path.MOVETO

# The codes will be all "LINETO" commands, except for "MOVETO"s at the beginning of each subpath

#spec = ((1, 1), (1, -1), (-1, 1), (-1, -1))
#spec = ((1, -1),)
spec = ((-1, 1),)
#spec = ((-1, -1),)

for i, (inside, outside) in enumerate(spec):
    # Concatenate the inside and outside subpaths together, changing their order as needed
    vertices = np.concatenate((outside_vertices[::outside],inside_vertices[::inside]))
    vertices[:, 0] += i * 2.5
    all_codes = np.concatenate((codes, codes))
    path = mpath.Path(vertices, all_codes)
    patch = mpatches.PathPatch(path, facecolor='#885500', edgecolor='black')
    ax.add_patch(patch)
    ax.annotate("Outside %s,\nInside %s" % (wise(outside), wise(inside)),(i * 2.5, -1.5), va="top", ha="center")
pass

ax.set_xlim(-2, 2.5*len(spec))
ax.set_ylim(-3, 2)
ax.set_title('Mmm, donuts!')
ax.set_aspect(1.0)
plt.show()


