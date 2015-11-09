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

np.set_printoptions(suppress=True, precision=3)

# sqlite3 database querying into ndarray
from  _npar import npar as q 

logging.basicConfig(level=logging.INFO)


o = np.load("/usr/local/env/dayabay/oxtorch/1.npy") 

flags = o.view(np.uint32)[:,3,3]
gflags_table(count_unique(flags))
brsa = maskflags_int("BR|SA|TORCH")

seq = np.load("/usr/local/env/dayabay/phtorch/1.npy") 
seqhis = seq[:,0,0]
seqmat = seq[:,0,1]

seqhis_table(count_unique(seqhis))
sqi = seqhis_int("TORCH BR SA")

#s = flags == brsa
s = seqhis == sqi        

rf = o[s]

r = np.sqrt(rf[:,0,0]*rf[:,0,0]+rf[:,0,1]*rf[:,0,1])
z = rf[:,0,2] - 300.
a = np.arctan(z/r)*180./np.pi


plt.hist(90.-a, bins=91, range=[0,90])



