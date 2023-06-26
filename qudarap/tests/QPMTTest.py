#!/usr/bin/env python

import os, numpy as np, matplotlib as mp
from opticks.ana.fold import Fold
import matplotlib.pyplot as plt
SIZE = np.array([1280, 720]) 
np.set_printoptions(edgeitems=16)

hc_eVnm = 1239.84198433200208455673 

e2w_ = lambda e:hc_eVnm/e
w2e_ = lambda w:hc_eVnm/w

PMTIDX = int(os.environ.get("PMTIDX","0")) # 0-based index into list of lpmtid 
SCRIPT = os.environ.get("SCRIPT", "unknown-SCRIPT")

class QPMTTest(object):

    NAMES = "NNVT HAMA NNVTHiQE".split()

    def __init__(self, t):
        self.t = t 
        self.init_energy_eV_domain() 
        self.init_mct_domain() 
        self.title_prefix = "%s : %s " % ( SCRIPT, t.base )

    def init_mct_domain(self):
        mct = self.t.qscan.mct_domain
        self.mct = mct 

    def init_energy_eV_domain(self):
        e = self.t.qscan.energy_eV_domain
        #e0,e1 = 2.3, 3.3
        e0,e1 = 1.55, 4.3
        w0,w1 = e2w_(e0), e2w_(e1)
        se = np.logical_and( e >= e0, e <= e1 ) 
        self.se = se 
        self.e0 = e0
        self.e1 = e1
        self.w0 = w0
        self.w1 = w1

    def present_lpmtcat_qeshape(self):
        t = self.t 
        se = self.se  
        e = t.energy_eV_domain
        a = t.qscan.lpmtcat_qeshape
        if a is None: return

        prop_ni = t.qpmt.qeshape[:,-1,-1].view(np.int32)  

        v0,v1 = 0.0,0.38

        assert len(a.shape) == 2, interp.shape 

        ni = a.shape[0]  # pmtcat
        nj = a.shape[1]  # energy

        title = "%s : qeshape GPU interpolation lines and values " % self.title_prefix

        fig, axs = mp.pyplot.subplots(1, ni, figsize=SIZE/100.)
        fig.suptitle(title)

        for i in range(ni):
            ax = axs[i]
            ax.set_ylim( v0, v1 )
            v = a[i] 
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


    def present_lpmtcat_rindex(self):
        t = self.t 

        a = t.qscan.lpmtcat_rindex
        if a is None: return 
        assert len(a.shape) == 4, a.shape 

        se = self.se  
        e = t.energy_eV_domain

        prop_ni = t.rindex[:,-1,-1].view(np.int32)  
        v0,v1 = -0.1,3.2

        ni = a.shape[0]  # pmtcat
        nj = a.shape[1]  # layers
        nk = a.shape[2]  # props

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
                    v = a[i,j,k]  
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


    def present_lpmtid_ART(self):
        """

        In [5]: t.lpmtid_ART.shape
        Out[5]: (9, 181, 4, 4)
        """

        t = self.t 
        lpmtid = t.qscan.lpmtid[PMTIDX] 

        a = t.qscan.lpmtid_ART
        if a is None: return 
        art = a[PMTIDX]
        mct = t.qscan.mct_domain

        consistent = len(art) == len(mct)

        if not consistent:
            log.error("present_lpmtid_ART : INCONSISTENT : art.shape %s mct.shape %s " % 
                     (str(art.shape), str(mct.shape)) ) 
            return
        pass
        assert consistent

        As   = art[...,0,0]
        Ap   = art[...,0,1]
        Aa   = art[...,0,2]
        A_   = art[...,0,3]

        Rs   = art[...,1,0]
        Rp   = art[...,1,1]
        Ra   = art[...,1,2]
        R_   = art[...,1,3]

        Ts   = art[...,2,0]
        Tp   = art[...,2,1]
        Ta   = art[...,2,2]
        T_   = art[...,2,3]

        SF    = art[...,3,0]
        wl    = art[...,3,1] 
        ARTa  = art[...,3,2]
        mct   = art[...,3,3]


        opt = os.environ.get("OPT", "A_,R_,T_,As,Rs,Ts,Ap,Rp,Tp,Aa,Ra,Ta")
        title = "%s : PMTIDX %d lpmtid %d OPT %s " % (t.base, PMTIDX, lpmtid, opt) 
        fig, ax = plt.subplots(1, figsize=SIZE/100.)
        fig.suptitle(title)

        if "As" in opt:ax.plot(  mct, As, label="As" )
        if "Ap" in opt:ax.plot(  mct, Ap, label="Ap" )
        if "Aa" in opt:ax.plot(  mct, Aa, label="Aa" )
        if "A_" in opt:ax.plot(  mct, A_, label="A_" )

        if "Rs" in opt:ax.plot(  mct, Rs, label="Rs" )
        if "Rp" in opt:ax.plot(  mct, Rp, label="Rp" )
        if "Ra" in opt:ax.plot(  mct, Ra, label="Ra" )
        if "R_" in opt:ax.plot(  mct, R_, label="R_" )

        if "Ts" in opt:ax.plot(  mct, Ts, label="Ts" )
        if "Tp" in opt:ax.plot(  mct, Tp, label="Tp" )
        if "Ta" in opt:ax.plot(  mct, Ta, label="Ta" )
        if "T_" in opt:ax.plot(  mct, T_, label="T_" )

        if "SF" in opt:ax.plot(  mct, SF, label="SF")
        if "wl" in opt:ax.plot(  mct, wl, label="wl" )
        if "ARTa" in opt:ax.plot(  mct, ARTa, label="ARTa" )
        if "mct" in opt:ax.plot(  mct, mct, label="mct" )


        ax.legend()
        fig.show()
            




        

    def check_lpmtcat(self):
        t = self.t 

        expect_lpmtcat = t.src_lcqs[t.lpmtid,0] 
        lpmtcat = t.lpmtid_stackspec[:,:,0,3].view(np.int32)    
        assert( np.all( lpmtcat[:,0] == expect_lpmtcat ) )

        lpmtid = t.lpmtid
        lpmtid_lpmtcat = np.max(t.lpmtid_stackspec[:,:,0,3].view(np.int32), axis=1)
        lpmtid_qe_scale = np.max(t.lpmtid_stackspec[:,:,1,3], axis=1)
        lpmtid_qe_shape = np.max(t.lpmtid_stackspec[:,:,2,3], axis=1)
        lpmtid_qe = np.max(t.lpmtid_stackspec[:,:,3,3], axis=1)

        expr = "np.c_[lpmtid,lpmtid_lpmtcat,lpmtid_qe_scale,lpmtid_qe_shape,lpmtid_qe]"
        lpmtid_tab = eval(expr)
        print("lpmtid_tab:%s\n%s" % ( expr,  lpmtid_tab))
        print(" note the qe_shape factor depends only on lpmtcat, the others have lpmtid dependency ") 
        print(" also note the qe_shape factor for lpmtcat 0:NNVT and 2:NNVT_HiQE are the same, diff from 1:HAMA  ")



if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    pt = QPMTTest(t)

    #plot = "lpmtcat_rindex"
    #plot = "lpmtcat_qeshape"
    plot = "lpmtid_ART"

    PLOT = os.environ.get("PLOT", plot )
    if PLOT == "lpmtcat_rindex":
        pt.present_lpmtcat_rindex()
    elif PLOT == "lpmtcat_qeshape":
        pt.present_lpmtcat_qeshape()
    elif PLOT == "lpmtid_ART":
        pt.present_lpmtid_ART()
    else:
        print("PLOT:%s not handled " % PLOT)
    pass



