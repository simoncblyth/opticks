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



 Spherical Coordinates (where theta is polar angle 0:pi, phi is azimuthal 0:2pi)

     x = r sin(th) cos(ph)  = r st cp   
     y = r sin(th) sin(ph)  = r st sp  
     z = r cos(th)          = r ct     


     sqrt(x*x + y*y) = r sin(th)
                  z  = r cos(th)

       atan( sqrt(x*x+y*y) / z ) = th 


* http://www.ece.rice.edu/~daniel/262/pdf/lecture13.pdf


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


oxs = np.logical_or(seqhis == sqi, seqhis == sqj)       
rxs = np.repeat(oxs, 10)

rf = ox[oxs]
sf = rx[rxs].reshape(-1,10,2,4)



xyz = rf[:,0,:3] - [0,0,300.]
r = np.linalg.norm(xyz, ord=2, axis=1) 
z = xyz[:,2]
zr = z/r
th = np.arccos(zr[zr<1])*180./np.pi

fig = plt.figure()

nx, ny = 2, 1

kwa = {}
kwa['bins'] = 91 
kwa['range'] = [0,90]
kwa['alpha'] = 0.5
kwa['log'] = False
kwa['histtype'] = "step"

ax = fig.add_subplot(ny,nx,1)
plt.plot(xd, spol, xd, ppol)


ax = fig.add_subplot(ny,nx,2)
ax.hist(th, **kwa)



fig.show()
