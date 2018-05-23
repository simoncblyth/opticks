#!/usr/bin/env python
"""

* https://matplotlib.org/2.0.0/examples/pylab_examples/contour_demo.html


"""
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.close()

delta = 0.1

x = np.arange(-4.0, 4.0, delta)
y = np.arange(-4.0, 4.0, delta)

X, Y = np.meshgrid(x, y)


Z = 2 - np.sqrt(X*X + Y*Y) 


plt.figure()



qcs = plt.contour(X, Y, Z)
plt.clabel(qcs, inline=1, fontsize=10)
plt.show()


