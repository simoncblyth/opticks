#!/usr/bin/env python
"""

    +       +



    +       +

"""

import numpy as np, logging, os
log = logging.getLogger(__name__)
from opticks.ana.view import View
import matplotlib.pyplot as plt

class FlightPath(object):
    """
    See optickscore/FlightPath.hh
    """
    FILENAME = "flightpath.npy"
    def __init__(self):
        self.views = []
  
    def as_array(self):
        log.info(" views %s " % len(self.views) )
        a = np.zeros( (len(self.views), 4, 4), dtype=np.float32 )
        for i, v in enumerate(self.views):
            a[i] = v.v
        pass
        return a 

    def save(self, dir_="/tmp"):
        path = os.path.join(dir_, self.FILENAME)
        a = fp.as_array()
        log.info("save %s to %s " % (repr(a.shape), path ))
        np.save(path, a ) 
        return a


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) 
    fp = FlightPath()

    look = [0., 0., 0.]

    tt = np.linspace( 0, 2*np.pi, 10 )[:-1]  # skip last to avoid repeating seam angle 
    xx = np.cos(tt)  
    yy = np.sin(tt)
    n = len(xx)

    for i in range(n): 
        j = (i+1)%n 
        print(i,j) 
        fp.views.append( View(eye=[xx[i],yy[i],0],look=[xx[j],yy[j],0]))
    pass

    a = fp.save()
    print a 


    plt.ion()
    fig = plt.figure(figsize=(6,5.5))
    plt.title("flightpath")
    ax = fig.add_subplot(111)
    sz = 3 
    ax.set_ylim([-sz,sz])
    ax.set_xlim([-sz,sz])
    ax.plot( a[:,0,0], a[:,0,1] ) 
    fig.show()








