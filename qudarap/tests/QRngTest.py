#!/usr/bin/env python
"""
::
 
    ipython -i QRngTest.py 

"""
import logging 
log = logging.getLogger(__name__)
import os,sys, numpy as np

try:
    import matplotlib.pyplot as plt 
except ImportError:
    plt = None
pass

np.set_printoptions(suppress=True, edgeitems=4, precision=5, linewidth=200 )


class QRngTest(object):
    FOLD = os.path.expandvars("$FOLD")
    def __init__(self, reldir):
        base = os.path.join(self.FOLD, reldir)

        upath = os.path.join(base, "u_0.npy")
        uupath = os.path.join(base, "uu.npy")

        u = np.load(upath) if os.path.exists(upath) else None
        uu = np.load(uupath) if os.path.exists(uupath) else None

        xtype = np.float32 if reldir.startswith("float") else np.float64

        if not u is None:assert(u.dtype == xtype)
        if not uu is None:assert(uu.dtype == xtype)

        self.reldir = reldir
        self.upath = upath
        self.uupath = uupath

        self.u = u
        self.uu = uu
        self.uuh = np.histogram( self.uu ) if not uu is None else None

        self.title = "qudarap/tests/QRngTest.py %s " % base


    def check_skipahead_shifts(self, offset):
        """
        For example when using skipaheadstep of 1::

           In [21]: np.all( uu[1,:,:-1] == uu[0,:,1:] )
           Out[21]: True

        The first dimension is the event index, so with skipahead per event
        of one expect to see the same randoms for each event but with a shift
        of one from the prior event. 

        """
        uu = self.uu
        assert len(uu.shape) == 3
        ni, nj, nk = uu.shape 
        for i in range(ni-1):
            i0 = i
            i1 = i+1
            assert np.all( uu[i1,:,:-offset] == uu[i0,:,offset:] )
            log.info("i0 %d i1 %d " % (i0,i1)) 
        pass

    def uu_plot(self):
        """
        Plot the uuh histogram : should be flat from 0 to 1 
        """
        if plt is None: return 
        t = self
        h = self.uuh 

        fig, ax = plt.subplots(figsize=[12.8,7.2])
        fig.suptitle(self.title)

        ax.plot( h[1][:-1], h[0], label="h", drawstyle="steps-post" )
        ax.set_ylim( 0, h[0].max()*1.1 )

        pass
        ax.legend()
        fig.show()
        path = os.path.join(t.FOLD, self.reldir, "fig.png")
        log.info("save to %s " % path)
        fig.savefig(path)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    TEST = os.environ.get("TEST", "generate") 

    reldir = "float/CHUNKED_CURANDSTATE"
    #reldir = "double"
    print("%s:TEST:%s reldir:%s" % ( sys.argv[0],TEST, reldir))

    t = QRngTest(reldir)  

    if TEST == "generate":
       u = t.u
       print("u.shape\n",u.shape)

    elif TEST == "generate_evid": 

       t.uu_plot()
       uu = t.uu
       uuh = t.uuh

       print("uu.shape\n",uu.shape)
       print("uu[:10]\n",uu[:10])
       print("uu[-10:]\n",uu[-10:])

       t.check_skipahead_shifts(1)
    else:
        print("%s:TEST:%s unhandled " % (sys.argv[0],TEST) )
    pass
pass


