#!/usr/bin/env python

import os, numpy as np
import opticks.ana.pvplt as pvp

if __name__ == '__main__':

    upos = np.random.rand(100000,3)
    unrm = np.random.rand(100000,3)

    pl = pvp.pvplt_plotter(label="pvplt_add_delta_lines_test.py")
    pvp.pvplt_add_points(pl, upos )
    pvp.pvplt_add_delta_lines(pl, upos[::1000], 0.1*unrm[::1000], color="r" )

    cp = pvp.pvplt_show(pl, incpoi=-5)
    




