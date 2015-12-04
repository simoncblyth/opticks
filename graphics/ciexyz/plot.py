#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from env.graphics.ciexyz.XYZ import XYZ


def plot_XYZ(xyz):
    fig = plt.figure()

    plt.plot(xyz.w,xyz.X,label="X", c="r")
    plt.plot(xyz.w,xyz.Y,label="Y", c="g")
    plt.plot(xyz.w,xyz.Z,label="Z", c="b")

    plt.legend()
    fig.show()

def plot_XYZr(xyz):
    fig = plt.figure()

    plt.plot(xyz.w,xyz.X,label="Xr", c="r")
    plt.plot(xyz.w,xyz.Y,label="Yr", c="g")
    plt.plot(xyz.w,xyz.Z,label="Zr", c="b")

    plt.legend()
    fig.show()

def plot_BB(xyz):
    fig = plt.figure()
    plt.plot(xyz.w,xyz.bb5k,label="5K", c="r")
    plt.plot(xyz.w,xyz.bb6k,label="6K", c="b")
    plt.legend()
    fig.show()


if __name__ == '__main__':

    w = np.linspace(300,800,501)

    plt.ion()

    xyz = XYZ()

    plot_XYZ(xyz)

    Xmax = xyz.w[np.argmax(xyz.X)]
    Ymax = xyz.w[np.argmax(xyz.Y)]
    Zmax = xyz.w[np.argmax(xyz.Z)]

    print "wavelengths (nm) of weighting function maxima X %s Y %s Z %s " % (Xmax, Ymax, Zmax)
    # Zmax looks magenta, not blue ?
    #


if 0:
    plot_XYZr(xyz)
    plot_BB(xyz)

    flat = np.ones_like(w)  # flat spectrum, will it be white ?  nope dull grey

    RGB = xyz.spectrumToRGB(flat)
    print "flat spectrum RGB:%s " % repr(RGB) 

    RGB = xyz.spectrumToRGB(xyz.bb5k)
    print "bb5k spectrum RGB:%s " % repr(RGB) 

    bb6k = xyz.bb6k
    RGB = xyz.spectrumToRGB(xyz.bb6k)
    print "bb6k spectrum RGB:%s " % repr(RGB) 




