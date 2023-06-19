#!/usr/bin/env python

import os, numpy as np, matplotlib as mp
from opticks.ana.fold import Fold

hc_eVnm = 1239.84198433200208455673 



e2w_ = lambda e:hc_eVnm/e
w2e_ = lambda w:hc_eVnm/w

SIZE = np.array([1280, 720]) 

class QPMTTest(object):
    def __init__(self, t):
        self.t = t 
    def present_interpolated_rindex(self):
        t = self.t 

        prop_ni = t.rindex[:,-1,-1].view(np.int32)  
        #names = t.rindex_names.lines 
        names = "NNVT HAMA NNVTHiQE".split()

        e = t.domain

        v0 = -0.1
        v1 = 3.2

        e0 = 2.3
        e1 = 3.3

        w0 = e2w_(e0)
        w1 = e2w_(e1)

        s = np.logical_and( e >= e0, e <= e1 ) 

        # (3, 4, 2, 1396)
        ni = t.interp.shape[0]  # pmtcat
        nj = t.interp.shape[1]  # layers
        nk = t.interp.shape[2]  # props


        title = "opticks/qudarap/tests/QPMTTest.sh : PMT layer refractive index interpolations on GPU  "

        fig, axs = mp.pyplot.subplots(1, ni, figsize=SIZE/100.)
        fig.suptitle(title)

        for i in range(ni):

            ax = axs[i]
            ax.set_ylim( v0, v1 )

            name = names[i]
            #ax.set_title(name)
            ax.set_xlabel("energy [eV]")

            sax = ax.secondary_xaxis('top', functions=(e2w_, w2e_))
            sax.set_xlabel('%s   wavelength [nm]' % name)
            # secondary_xaxis w2e_ : RuntimeWarning: divide by zero encountered in true_divide

            for j in range(nj):
                if j in [0,3]: continue   # skip layers 0,3 Pyrex,Vacuum 
                for k in range(nk):
                    v = t.interp[i,j,k]  
                    iprop = i*nj*nk + j*nk + k 

                    label = "L%d %sINDEX" % ( j, "R" if k == 0 else "K" )

                    ax.plot( e[s], v[s], label=label ) 

                    p_ni = prop_ni[iprop]
                    p_e = t.rindex[iprop,:p_ni,0] 
                    p_v = t.rindex[iprop,:p_ni,1] 

                    p_s = np.logical_and( p_e >= e0, p_e <= e1 )
                    ax.scatter( p_e[p_s], p_v[p_s] )
                pass
            pass
            ax.legend(loc=os.environ.get("LOC", "lower right")) # upper/center/lower right/left 
        pass
        fig.show()





if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    pt = QPMTTest(t)
    pt.present_interpolated_rindex()



