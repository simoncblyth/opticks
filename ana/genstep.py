#!/usr/bin/env python
"""
genstep.py: Fit genstep xyz vs time, used for viewpoint tracking 
=================================================================

Fit the genstep xyz vs time to obtain parametric eqn of genstep position 
with time parameter.::

    In [3]: tk = A.load_("cerenkov","1_track","dayabay")

    In [4]: tk
    Out[4]: A(cerenkov,1_track,dayabay)

    In [5]: print tk
    [[ -16390.518 -802295.938   -7059.101]
     [   -162.573     251.993       0.172]]

::

    In [1]: run genstep.py
    [[[   0.177   -1.583    4.94     1.   ]
      [-252.339  -45.677 -155.278    0.   ]
      [   0.      82.83     0.       0.   ]]]
    INFO:opticks.ana.nload:saving derivative of A(cerenkov,1,juno) to /usr/local/env/opticks/juno/cerenkov/1_track.npy 

::

    simon:npy blyth$ /usr/local/opticks.ana/bin/NumpyEvtTest
    [2016-Mar-25 12:08:16.965734]:info: NumpyEvt::loadGenstepDerivativeFromFile   typ cerenkov tag 1_track det dayabay
    [2016-Mar-25 12:08:16.966326]:info: NumpyEvt::loadGenstepDerivativeFromFile (3,4) 

    (  0)  -16390.518  -802295.938   -7059.101       1.000 
    (  1)    -162.573     251.993       0.172       0.000 
    (  2)       0.844      27.423       0.000       0.000 

"""
import os, logging
import numpy as np
import matplotlib.pyplot as plt

from opticks.ana.base import opticks_main
from opticks.ana.nload import A, I, II, path_

log = logging.getLogger(__name__)

X,Y,Z,W,T = 0,1,2,3,3


if __name__ == '__main__':

    args = opticks_main(det="juno", src="cerenkov", tag="1")

    try:    
        a = A.load_("gensteps",args.src,args.tag,args.det)
    except IOError as err:
        log.fatal(err)
        sys.exit(args.mrc)


    log.info("loaded gensteps %s %s %s " % (a.path, a.stamp, repr(a.shape)))

    #path = os.path.expandvars("$LOCAL_BASE/opticks/opticksdata/gensteps/dayabay/cerenkov/1.npy")
    #path = os.path.expandvars("$LOCAL_BASE/opticks/opticksdata/gensteps/juno/cerenkov/1.npy")
    #a = np.load(path)


    xyzt = a[:,1]
    #print xyzt
    x,y,z,t = xyzt[:,X], xyzt[:,Y], xyzt[:,Z], xyzt[:,T]

    tr = [t.min(), t.max()]
    xr = [x.min(), x.max()]
    yr = [y.min(), y.max()]
    zr = [z.min(), z.max()]


    plt.close()
    plt.ion()
    ny,nx = 1,3

    fig = plt.figure()

    ax = fig.add_subplot(ny,nx,1)
    ax.scatter(t, x)
    xf = np.polyfit(t,x,1,full=True)
    xm, xc = xf[0]
    xl = [xm*tt + xc for tt in tr]
    ax.plot(tr, xl, '-r')

    ax = fig.add_subplot(ny,nx,2)
    ax.scatter(t, y)
    yf = np.polyfit(t,y,1,full=True)
    ym, yc = yf[0]
    yl = [ym*tt + yc for tt in tr]
    ax.plot(tr, yl, '-r')

    ax = fig.add_subplot(ny,nx,3)
    ax.scatter(t, z)
    zf = np.polyfit(t,z,1,full=True)
    zm, zc = zf[0]
    zl = [zm*tt + zc for tt in tr]
    ax.plot(tr, zl, '-r')


    track = np.array(
                [[
                  [xc,yc,zc,1.0],
                  [xm,ym,zm,0.0],
                  [tr[0],tr[1],0.0,0.0]
                ]], dtype=np.float32)
    print track


    #a.derivative_save(track, "track")




