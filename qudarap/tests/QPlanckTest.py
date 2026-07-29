#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
import matplotlib.pyplot as mp
SIZE=np.array([1280, 720])
TEST = os.environ.get("TEST","")


def planck_sample_cuda(f):
    sample = f.wavelength
    print(f"planck_sample_cuda sample.shape {sample.shape}")
    fig, ax = mp.subplots(figsize=SIZE/100.)
    title= f"QPlanckTest.sh : {f.base} : {TEST}  "
    fig.suptitle(title)

    w = np.arange(80.,801,.1, dtype=np.float64)
    wbin = w[::100]

    hn, hd = np.histogram(sample, wbin)
    assert np.all(hd == wbin)
    assert len(hn) == len(wbin) - 1   # looses one bin

    ax.plot( hd[:-1], hn , drawstyle="steps-post", label="sample")  # -pre -mid -post

    ax.legend()
    fig.show()




if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))
    planck_sample_cuda(f)
pass
