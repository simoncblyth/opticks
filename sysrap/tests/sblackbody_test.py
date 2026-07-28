#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
import matplotlib.pyplot as mp
SIZE=np.array([1280, 720])
TEST = os.environ.get("TEST","")


def planck_spectral_radiance_set(f):
    psr = f.planck_spectral_radiance_set
    psr_names = f.planck_spectral_radiance__set_names

    print(psr)
    print(psr_names)

    fig, ax = mp.subplots(figsize=SIZE/100.)
    title= f"sblackbody_test.sh : {f.base} : {TEST}  "
    fig.suptitle(title)
    for i in range(len(psr)):
        nm = psr[i,:,0]
        bb = psr[i,:,1]
        bb /= bb.sum()
        ax.plot( nm, bb, label=psr_names[i] )
    pass
    ax.legend()
    fig.show()


def planck_cdf(f):
    cdf = f.cdf
    print(f"planck_cdf cdf.shape {cdf.shape}")
    fig, ax = mp.subplots(figsize=SIZE/100.)
    title= f"sblackbody_test.sh : {f.base} : {TEST}  "
    fig.suptitle(title)
    ax.plot( cdf[:,0], cdf[:,1], label="cdf" )
    ax.legend()
    fig.show()

def planck_icdf_0(f):
    icdf = f.icdf.reshape(-1)
    print(f"planck_icdf_0 icdf.shape {icdf.shape}")
    fig, ax = mp.subplots(figsize=SIZE/100.)
    title= f"sblackbody_test.sh : {f.base} : {TEST}  "
    fig.suptitle(title)

    idom = np.linspace(0,1, len(icdf) )
    ax.plot( idom, icdf, label="icdf" )
    ax.legend()
    fig.show()

def planck_icdf(f):
    icdf_prop = f.icdf_prop
    print(f"planck_icdf icdf_prop.shape {icdf_prop.shape}")
    fig, ax = mp.subplots(figsize=SIZE/100.)
    title= f"sblackbody_test.sh : {f.base} : {TEST}  "
    fig.suptitle(title)

    ax.plot( icdf_prop[:,0], icdf_prop[:,1], label="icdf_prop" )
    ax.legend()
    fig.show()

def planck_sample(f):
    sample = f.wavelength
    print(f"planck_sample sample.shape {sample.shape}")
    fig, ax = mp.subplots(figsize=SIZE/100.)
    title= f"sblackbody_test.sh : {f.base} : {TEST}  "
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

    if TEST == "planck_spectral_radiance_set":
        planck_spectral_radiance_set(f)
    elif TEST == "planck_cdf":
        planck_cdf(f)
    elif TEST == "planck_icdf":
        planck_icdf(f)
    elif TEST == "planck_sample":
        planck_sample(f)
    else:
        print(f"TEST {TEST} unhandled")
    pass


