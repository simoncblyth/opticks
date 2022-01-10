#!/usr/bin/env python
"""

"""
import numpy as np
import pyvista as pv
pv = None

import matplotlib.pyplot as mp                                                                                                                                                                                         

if __name__ == '__main__':

    path = os.path.expandvars(os.path.join("/tmp/$USER/opticks/CSGTargetGlobalTest/$MOI", "ce.npy")) 
    ce = np.load(path)

    SIZE = np.array([1280, 720])


    moi = os.environ.get("MOI","-")

    print("moi %s " % moi )
    print("path %s " % path )
    print("ce.shape %s " % str(ce.shape))

    r = np.sqrt(np.sum(ce[:,:3]*ce[:,:3], axis=1))                                                                                                                                                    
    rmin = r.min()
    rmax = r.max()
    print("rmin %s rmax %s rmax-rmin %s " % (rmin, rmax, rmax-rmin))


    if not pv is None:
        pl = pv.Plotter(window_size=2*SIZE)
        pl.add_points(ce[:,:3])
        pl.show_grid()
        cp = pl.show()                                   
    pass

    from j.PosFile.RadiusTest import RadiusTest
    rt = RadiusTest()



    title = [
          "opticks/CSG/tests/CSGTargetGlobalTest.sh MOI %s" % moi, 
          "radii from axis aligned CE center-extents (wobbles expected as using global frame CE)" 
          ]

    if not mp is None:
        fig, ax = mp.subplots(figsize=SIZE/100.)
        fig.suptitle("\n".join(title))

        i = range(len(r))
        ax.scatter( i, r )
        xlim = ax.get_xlim() 

        kk = "fastener_r xjfixture_r xjanchor_r sjfixture_r sjreceiver_fastener_r".split()

        rr = list(map(lambda k:rt.d[k], kk))
        for i in range(len(kk)):
            ax.plot( xlim, [rr[i],rr[i]], label=kk[i] ) 
        pass
        ax.legend()
        fig.show()
        outpath = os.path.expandvars("/tmp/$USER/opticks/CSG/tests/CSGTargetGlobalTest/${MOI}_radii.png")
        outdir = os.path.dirname(outpath)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        pass
        print("save to %s " % outpath)
        fig.savefig(outpath)
    pass
  

