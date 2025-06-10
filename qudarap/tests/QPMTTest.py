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
        self.init_theta_radians_domain()
        self.init_mct_domain()
        self.init_costh_domain()
        self.title_prefix = "%s : %s " % ( SCRIPT, t.base )

    def init_mct_domain(self):
        mct = self.t.qscan.mct_domain
        self.mct = mct

    def init_energy_eV_domain(self):
        e = self.t.qscan.energy_eV_domain
        #e0,e1 = 2.3, 3.3
        e0,e1 = 1.55, 4.3
        w0,w1 = e2w_(e0), e2w_(e1)
        ese = np.logical_and( e >= e0, e <= e1 )

        self.e0 = e0
        self.e1 = e1
        self.w0 = w0
        self.w1 = w1
        self.ese = ese

    def init_theta_radians_domain(self):
        h = self.t.qscan.theta_radians_domain
        h0 = h[0]
        h1 = h[-1]
        hse = np.logical_and( h >= h0, h <= h1 )

        self.h0 = h0
        self.h1 = h1
        self.hse = hse

    def init_costh_domain(self):
        c = self.t.qscan.costh_domain
        c0 = c[-1]   ## flip the order as reversed
        c1 = c[0]
        cse = np.logical_and( c >= c0, c <= c1 )

        self.c0 = c0
        self.c1 = c1
        self.cse = cse



    def present_qeshape(self):
        t = self.t
        se = self.ese
        d = t.qscan.energy_eV_domain
        a = t.qscan.lpmtcat_qeshape
        if a is None: return

        prop_ni = t.qpmt.qeshape[:,-1,-1].view(np.int32)  ## last values from input prop arrays

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
            ax.plot( d[se], v[se], label=label )
            ax.legend(loc=os.environ.get("LOC", "upper left")) # upper/center/lower right/left

            p_e = t.qpmt.qeshape[i,:prop_ni[i],0]
            p_v = t.qpmt.qeshape[i,:prop_ni[i],1]
            p_s = np.logical_and( p_e >= self.e0, p_e <= self.e1 )

            ax.scatter( p_e[p_s], p_v[p_s] )
        pass
        fig.show()


    def present_cetheta(self):
        t = self.t
        se = self.hse
        d = t.qscan.theta_radians_domain
        a = t.qscan.lpmtcat_cetheta
        if a is None: return

        prop_ni = t.qpmt.cetheta[:,-1,-1].view(np.int32)  ## last values from input prop arrays

        v0,v1 = 0.0,1.1

        assert len(a.shape) == 2, interp.shape

        ni = a.shape[0]  # pmtcat
        nj = a.shape[1]  # theta

        title = "%s : cetheta GPU interpolation lines and values " % self.title_prefix

        fig, axs = mp.pyplot.subplots(1, ni, figsize=SIZE/100.)
        fig.suptitle(title)

        for i in range(ni):
            ax = axs[i]
            ax.set_ylim( v0, v1 )
            v = a[i]
            name = self.NAMES[i]
            label = "%s cetheta" % name
            ax.set_xlabel("theta [radians]")
            ax.plot( d[se], v[se], label=label )
            ax.legend(loc=os.environ.get("LOC", "upper left")) # upper/center/lower right/left

            ## input (domain,value) pairs used by the interpolation
            p_d = t.qpmt.cetheta[i,:prop_ni[i],0]
            p_v = t.qpmt.cetheta[i,:prop_ni[i],1]
            p_s = np.logical_and( p_d >= self.h0, p_d <= self.h1 )

            ax.scatter( p_d[p_s], p_v[p_s] )
        pass
        fig.show()


    def present_cecosth(self):
        t = self.t
        se = self.cse
        d = t.qscan.costh_domain
        a = t.qscan.lpmtcat_cecosth
        if a is None: return

        prop_ni = t.qpmt.cecosth[:,-1,-1].view(np.int32)  ## last values from input prop arrays

        v0,v1 = 0.0,1.1

        assert len(a.shape) == 2, interp.shape

        ni = a.shape[0]  # pmtcat
        nj = a.shape[1]  # theta

        title = "%s : cecosth GPU interpolation lines and values " % self.title_prefix

        fig, axs = mp.pyplot.subplots(1, ni, figsize=SIZE/100.)
        fig.suptitle(title)

        for i in range(ni):
            ax = axs[i]
            ax.set_ylim( v0, v1 )
            v = a[i]
            name = self.NAMES[i]
            label = "%s cecosth" % name
            ax.set_xlabel("cosine_theta")
            ax.plot( d[se], v[se], label=label )
            ax.legend(loc=os.environ.get("LOC", "upper left")) # upper/center/lower right/left

            ## input (domain,value) pairs used by the interpolation
            p_d = t.qpmt.cecosth[i,:prop_ni[i],0]
            p_v = t.qpmt.cecosth[i,:prop_ni[i],1]
            p_s = np.logical_and( p_d >= self.c0, p_d <= self.c1 )

            ax.scatter( p_d[p_s], p_v[p_s] )
        pass
        fig.show()





    def present_rindex(self):
        t = self.t

        a = t.qscan.lpmtcat_rindex
        if a is None: return
        assert len(a.shape) == 4, a.shape

        se = self.se
        e = t.qscan.energy_eV_domain

        prop_ni = t.qpmt.rindex[:,-1,-1].view(np.int32)
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
                    p_e = t.qpmt.rindex[iprop,:p_ni,0]
                    p_v = t.qpmt.rindex[iprop,:p_ni,1]

                    p_s = np.logical_and( p_e >= self.e0, p_e <= self.e1 )
                    ax.scatter( p_e[p_s], p_v[p_s] )
                pass
            pass
            ax.legend(loc=os.environ.get("LOC", "lower right")) # upper/center/lower right/left
        pass
        fig.show()


    def present_atqc(self):
        """
        In [4]: t.qscan.atqc.shape
        Out[4]: (9, 900, 4)

        In [5]: t.qscan.lpmtid
        Out[5]: array([    0,    10,    55,    98,   100,   137,  1000, 10000, 17611], dtype=int32)

        t.qscan.mct_domain.shape
        Out[8]: (900,)

        900 is scan over mct : minus_cos_theta of angle of incidence   (NOT landing position) 

        atqc[:,:,3]
              behaves as expected, fixed across the scan at high values ~0.91 - 0.97 depending on pmtid 

        """
        pass
        t = self.t
        mct = t.qscan.mct_domain
        atqc = t.qscan.atqc



    def present_art(self):
        """
        In [2]: t.qscan.art.shape
        Out[2]: (9, 181, 4, 4)
        """

        t = self.t
        lpmtid = t.qscan.lpmtid[PMTIDX]   # pick lpmtid by PMTIDX

        all_art = t.qscan.art
        if all_art is None:
            print("present_art : ABORT t.qscan.art is None ")
            return
        pass
        art = all_art[PMTIDX]
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

    #plot = "rindex"
    #plot = "qeshape"
    #plot = "cetheta"
    plot = "cecosth"
    #plot = "art"

    PLOT = os.environ.get("PLOT", plot )
    if PLOT == "rindex":
        pt.present_rindex()
    elif PLOT == "qeshape":
        pt.present_qeshape()
    elif PLOT == "cetheta":
        pt.present_cetheta()
    elif PLOT == "cecosth":
        pt.present_cecosth()
    elif PLOT == "art":
        pt.present_art()
    else:
        print("PLOT:%s not handled " % PLOT)
    pass



