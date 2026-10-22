#!/usr/bin/env python
"""
S4MultiLayrStack_test2.py
===========================

Compare the ART calculation from S4MultiLayrStack.h with that 
from the python package called "tmm". 

* https://github.com/sbyrnes321/tmm
* https://pypi.org/project/tmm/

coh_tmm returns a dict, so it cannot be used in 
a numpy efficient vectorized way.

Effecting the flip so can scan from 0 to 180 degrees
and flipping for the 2nd half. Follow C4MultiLayrStack.h::

    l0.ct = minus_cos_theta < zero  ? -mct : mct ; 
    //
    //  flip picks +ve ct that constrains the angle to first quadrant 
    //  this works as : cos(pi-theta) = -cos(theta)
    //  without flip, the ART values are non-physical : always outside 0->1 for mct > 0 angle > 90  
    //

"""
import numpy as np
from np.fold import Fold
from tmm.tmm_core import coh_tmm 
np.set_printoptions(precision=10)  


def test_stack(a):
    """
    """
    ss  = a[:, 0]
    art = a[:, 1]
    art2 = np.zeros_like( art )

    SF    = art[:,3,0]
    wl    = art[:,3,1] 
    dpcmn = art[:,3,2] 
    mct   = art[:,3,3]
    stst = 1. - mct*mct  # zero for mct == 1/-1
    SF2 = np.zeros_like(stst)   
    w = np.where( stst > 0. )
    SF2[w] = dpcmn[w]*dpcmn[w]/stst[w]


    # for mct from -1 to 0 : aoi increases from 0 to pi/2
    # for mct from  0 to 1 : aoi increases from pi/2 to pi 
    #  so do a flip 
    ct = np.abs(mct)     # equivalent to : -mct if mct < 0 else mct (for the flip)
    aoi = np.arccos(ct)  # normal:0 glancing:np.pi/2

    # prep inputs needed by tmm
    d_list = ss[:,:,2]
    d_list[np.where(d_list==0.)] = np.inf  
    n_list = ss[:,:,0] + 1j*ss[:,:,1]

    ds = np.empty([len(ss)], dtype=np.object )   
    dp = np.empty([len(ss)], dtype=np.object )   

    # forced to do a slow python loop as pypi tmm is not implemented in vectorizable way  
    for i in range(len(ss)):

        if mct[i] < 0.:
            ds[i] = coh_tmm('s', n_list[i], d_list[i], aoi[i], wl[i] )
            dp[i] = coh_tmm('p', n_list[i], d_list[i], aoi[i], wl[i] ) 
        else:
            ds[i] = coh_tmm('s', n_list[i][::-1], d_list[i][::-1], aoi[i], wl[i] )
            dp[i] = coh_tmm('p', n_list[i][::-1], d_list[i][::-1], aoi[i], wl[i] ) 
        pass

        A_s = 1. - ds[i]['R'] - ds[i]['T']
        A_p = 1. - dp[i]['R'] - dp[i]['T'] 

        R_s = ds[i]['R']
        R_p = dp[i]['R']

        T_s = ds[i]['T']
        T_p = dp[i]['T']

        A = A_s*SF[i] + A_p*(1.-SF[i]) 
        R = R_s*SF[i] + R_p*(1.-SF[i]) 
        T = T_s*SF[i] + T_p*(1.-SF[i]) 

        art2[i,0,0] = A_s
        art2[i,0,1] = A_p
        art2[i,0,2] = (A_s + A_p)/2.
        art2[i,0,3] = A

        art2[i,1,0] = R_s
        art2[i,1,1] = R_p
        art2[i,1,2] = (R_s + R_p)/2.
        art2[i,1,3] = R

        art2[i,2,0] = T_s
        art2[i,2,1] = T_p
        art2[i,2,2] = (T_s + T_p)/2.
        art2[i,2,3] = T
    pass
    dart = art[:,:3] - art2[:,:3]   
    print(dart.max())
    return dart, art, art2




if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    if not getattr(f, "aoi", None) is None:
        a = f.aoi
        print("py:test_stack_aoi a:%s " % str(a.shape))
        assert len(a.shape) == 4 
        dart, art, art2 = test_stack( a )
        assert dart.max() < 1e-7
    pass

    if not getattr(f, "aoi_pol", None) is None:
        b = f.aoi_pol 
        assert len(b.shape) == 5
        c = b.reshape( b.shape[0]*b.shape[1], b.shape[2], b.shape[3], b.shape[4] )

        print("py:test_stack_aoi_pol b:%s c:%s " % (str(b.shape), str(c.shape) ))

        #
        # Note how it is convenient to use different shapes at 
        # different junctures... here its more convenient
        # to flatten the top two aoi and pol dimensions 
        # down to a single "item" dimension  : so can reuse test_stack 
        #
        py_test = True    # pypi tmm is real slow
        #py_test = False
        if py_test:
            assert len(c.shape) == 4
            dart,art,art2 = test_stack(c)
            assert dart.max() < 1e-7
        pass

 
        SF = c[:,1,3,0] 
        wl = c[:,1,3,1] 
        dpcmn = c[:,1,3,2] 
        mct = c[:,1,3,3]

        assert len( np.unique(wl)) == 1 

        assert SF.min() >= 0. 
        assert SF.max() <= 1.

        assert dpcmn.min() >= -1. 
        assert dpcmn.max() <= 1.

        assert mct.min() >= -1. 
        assert mct.max() <= 1.
    pass
pass


