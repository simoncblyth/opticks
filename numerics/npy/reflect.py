#!/usr/bin/env python
"""
Reflection distrib following BoxInBox with save::

    ggv-bib --save

    ggv.sh --test \
           --eye 0.5,0.5,0.0 \
           --animtimemax 7 \
           --testconfig "mode=BoxInBox_dimensions=500,300,0,0_boundary=Rock//perfectAbsorbSurface/Vacuum_boundary=Vacuum///Pyrex_" \
           --torchconfig "frame=1_type=invsphere_source=0,0,300_target=0,0,0_radius=102_zenithazimuth=0,0.5,0,1_material=Vacuum" \
            $*


                    + [x,y,z-300]
                   /|
                  / |
                 /  | 
                /   |
    -----------+----+--------
        [0,0,300] r 


TODO:

* compare to Fresnel eqn
* refractive index lookup
* distinguish S and P 

"""
import os, logging
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from env.python.utils import *
from env.numerics.npy.types import *
import env.numerics.npy.PropLib as PropLib 


np.set_printoptions(suppress=True, precision=3)

# sqlite3 database querying into ndarray
from  _npar import npar as q 

logging.basicConfig(level=logging.INFO)


mlib = PropLib.PropLib("GMaterialLib")

wavelength = 500 
m1 = "Vacuum"
m2 = "Pyrex"

n1 = mlib.interp(m1,wavelength,PropLib.M_REFRACTIVE_INDEX)
n2 = mlib.interp(m2,wavelength,PropLib.M_REFRACTIVE_INDEX)
xd = np.linspace(0,90,91)
x = xd*np.pi/180.

def fresnel(x, n1, n2, spol=True):
    """
    https://en.wikipedia.org/wiki/Fresnel_equations
    """
    cx = np.cos(x)
    sx = np.sin(x) 
    disc = 1. - np.square(n1*sx/n2)
    qdisc = np.sqrt(disc)
    pass
    if spol:
        num = (n1*cx - n2*qdisc)
        den = (n1*cx + n2*qdisc) 
    else:
        num = (n1*qdisc - n2*cx)
        den = (n1*qdisc + n2*cx) 
    return np.square(num/den) 


spol = fresnel(x, n1, n2, True)
ppol = fresnel(x, n1, n2, False)

src = "torch"
tag = "1"

ox = load_("ox"+src,tag) 
rx = load_("rx"+src,tag) 
ph = load_("ph"+src,tag)

seqhis = ph[:,0,0]
seqmat = ph[:,0,1]

flags = ox.view(np.uint32)[:,3,3]
gflags_table(count_unique(flags))
brsa = maskflags_int("BR|SA|TORCH")


seqhis_table(count_unique(seqhis))
sqi = seqhis_int("TORCH BR SA")
sqj = seqhis_int("TORCH BR AB")

# hmm misses reflected photons absorbed before hitting the sides


#s = flags == brsa
s = np.logical_or(seqhis == sqi, seqhis == sqj)       

rf = ox[s]


r = np.sqrt(rf[:,0,0]*rf[:,0,0]+rf[:,0,1]*rf[:,0,1])
z = rf[:,0,2] - 300.

a = 90. - np.arctan(z/r)*180./np.pi

ap = a[rf[:,0,0]>0]
am = a[rf[:,0,0]<0]


fig = plt.figure()

nx, ny = 3, 1

kwa = {}
kwa['bins'] = 91 
kwa['range'] = [0,90]
kwa['alpha'] = 0.5
kwa['log'] = False


ax = fig.add_subplot(ny,nx,1)
plt.plot(xd, spol, xd, ppol)

ax = fig.add_subplot(ny,nx,2)
ax.hist(ap, **kwa)

ax = fig.add_subplot(ny,nx,3)
ax.hist(am, **kwa)


fig.show()
