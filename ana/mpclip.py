#!/usr/bin/env python
"""
https://stackoverflow.com/questions/10192011/clipping-in-matplotlib-why-doesnt-this-work

"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


fig, ax = plt.subplots(1)

rect = Rectangle((-2,-2),4,2, facecolor="none", edgecolor="none")
circle = Circle((0,0),1)

ax.add_artist(rect)      ## commenting this makes nothing appear 
ax.add_artist(circle)

circle.set_clip_path(rect)

plt.axis('equal')
plt.axis((-2,2,-2,2))

fig.show()

