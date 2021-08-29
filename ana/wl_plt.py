#!/usr/bin/env python
"""
wl_plt.py
============

While using::

    tds3gun.sh 

Run this to plot the "a" and "b" events provided by ab.py::

    In [3]: run wl_plt.py

    In [4]: wl_plt(a,b)

    In [5]: a.sel = "CK .."      ## select all history categories starting "CK" 

    In [6]: b.sel = "CK .."

    In [7]: wl_plt(a,b)

"""
import numpy as np 
import matplotlib.pyplot as plt 

def wl_plt(a, b):

    dom = np.arange(80, 800, 50)

    ah = np.histogram( a.ox[:,2,3], dom)
    bh = np.histogram( b.ox[:,2,3], dom)

    fig, ax = plt.subplots()
    ax.plot( dom[:-1], ah[0], label="a", drawstyle="steps-post" )
    ax.plot( dom[:-1], bh[0], label="b", drawstyle="steps-post"  )
    ax.legend()
    fig.show()


def wl_plt_ck(a, b):
    a.sel = "CK .."
    b.sel = "CK .."
    wl_plt(a, b)

def wl_plt_si(a, b):
    a.sel = "SI .."
    b.sel = "SI .."
    wl_plt(a, b)



