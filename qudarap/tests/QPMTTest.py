#!/usr/bin/env python

import os, numpy as np, matplotlib as mp
from opticks.ana.fold import Fold

hc_eVnm = 1239.84198433200208455673 

e2w_ = lambda e:hc_eVnm/e
w2e_ = lambda w:hc_eVnm/w

SIZE = np.array([1280, 720]) 
SCRIPT = os.environ.get("SCRIPT", "unknown-SCRIPT")

class QPMTTest(object):

    NAMES = "NNVT HAMA NNVTHiQE".split()

    def __init__(self, t):
        self.t = t 
        e = t.domain
        #e0,e1 = 2.3, 3.3
        e0,e1 = 1.55, 4.3
        w0,w1 = e2w_(e0), e2w_(e1)

        se = np.logical_and( e >= e0, e <= e1 ) 

        self.se = se 
        self.e0 = e0
        self.e1 = e1
        self.w0 = w0
        self.w1 = w1
        self.title_prefix = "%s : %s " % ( SCRIPT, t.base )


    def present_qeshape_interp(self):
        t = self.t 
        se = self.se  
        e = t.domain

        interp = t.qeshape_interp  
        print(interp.shape)

        prop_ni = t.qeshape[:,-1,-1].view(np.int32)  

        v0,v1 = 0.0,0.38

        assert len(interp.shape) == 2, interp.shape 

        ni = interp.shape[0]  # pmtcat
        nj = interp.shape[1]  # energy

        title = "%s : qeshape GPU interpolation lines and values " % self.title_prefix

        fig, axs = mp.pyplot.subplots(1, ni, figsize=SIZE/100.)
        fig.suptitle(title)

        for i in range(ni):
            ax = axs[i]
            ax.set_ylim( v0, v1 )
            v = interp[i] 
            name = self.NAMES[i]
            label = "%s qeshape" % name 
            ax.set_xlabel("energy [eV]")
            ax.plot( e[se], v[se], label=label ) 
            ax.legend(loc=os.environ.get("LOC", "upper left")) # upper/center/lower right/left 

            p_e = t.qeshape[i,:prop_ni[i],0] 
            p_v = t.qeshape[i,:prop_ni[i],1] 
            p_s = np.logical_and( p_e >= self.e0, p_e <= self.e1 )

            ax.scatter( p_e[p_s], p_v[p_s] )
        pass
        fig.show()


    def present_rindex_interp(self):
        t = self.t 
        se = self.se  
        e = t.domain

        prop_ni = t.rindex[:,-1,-1].view(np.int32)  

        v0,v1 = -0.1,3.2

        interp = t.rindex_interp  # (3, 4, 2, 1396)

        assert len(interp.shape) == 4, interp.shape 

        ni = interp.shape[0]  # pmtcat
        nj = interp.shape[1]  # layers
        nk = interp.shape[2]  # props

        title = "%s : PMT layer refractive index interpolations on GPU  " % self.title_prefix

        fig, axs = mp.pyplot.subplots(1, ni, figsize=SIZE/100.)
        fig.suptitle(title)

        for i in range(ni):

            ax = axs[i]
            ax.set_ylim( v0, v1 )

            name = self.NAMES[i]
            #ax.set_title(name)
            ax.set_xlabel("energy [eV]")

            sax = ax.secondary_xaxis('top', functions=(e2w_, w2e_))
            sax.set_xlabel('%s   wavelength [nm]' % name)
            # secondary_xaxis w2e_ : RuntimeWarning: divide by zero encountered in true_divide

            for j in range(nj):
                if j in [0,3]: continue   # skip layers 0,3 Pyrex,Vacuum 
                for k in range(nk):
                    v = interp[i,j,k]  
                    iprop = i*nj*nk + j*nk + k 

                    label = "L%d %sINDEX" % ( j, "R" if k == 0 else "K" )

                    ax.plot( e[se], v[se], label=label ) 

                    p_ni = prop_ni[iprop]
                    p_e = t.rindex[iprop,:p_ni,0] 
                    p_v = t.rindex[iprop,:p_ni,1] 

                    p_s = np.logical_and( p_e >= self.e0, p_e <= self.e1 )
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

    PLOT = os.environ.get("PLOT", "rindex")
    if PLOT == "rindex":
        pt.present_rindex_interp()
    elif PLOT == "qeshape":
        pt.present_qeshape_interp()
    else:
        print("PLOT:%s not handled " % PLOT)
    pass



