#!/usr/bin/env python
"""

https://stackoverflow.com/questions/33551103/plotting-a-sphere-mesh-with-matplotlib

In [5]: (x.shape,y.shape,z.shape)
Out[5]: ((13, 7), (13, 7), (13, 7))



http://matplotlib.org/1.2.1/examples/mplot3d/trisurf3d_demo.html


"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    plt.ion()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 13)
    v = np.linspace(0, np.pi, 7)

    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='w', shade=0)

    plt.show()
