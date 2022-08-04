#!/usr/bin/env python

import numpy as np
from numpy.linalg import multi_dot

from opticks.ana.fold import Fold
from opticks.CSG.CSGFoundry import CSGFoundry 
from opticks.ana.eprint import eprint, epr
from opticks.sysrap.stree import stree



def check_inverse_pair(f, a_="inst", b_="iinst" ):
    """
    :param f: Fold instance
     
    CAUTION : this clears the identity info prior to checking transforms
    """
    a = getattr(f, a_)
    b = getattr(f, b_)

    a[:,:,3] = [0.,0.,0.,1.]
    b[:,:,3] = [0.,0.,0.,1.]

    assert len(a) == len(b)

    chk = np.zeros( (len(a), 4, 4) )   
    for i in range(len(a)): chk[i] = np.dot( a[i], b[i] )   
    chk -= np.eye(4)

    print("check_inverse_pair :  %s %s " % (a_, b_))
    print(chk.min(), chk.max())


def compare_f_with_cf(f, cf ):
    print("compare_f_with_cf")
    eprint("(cf.inst - f.inst_f4).max()", globals(), locals())  
    eprint("(cf.inst - f.inst_f4).min()", globals(), locals())  


if __name__ == '__main__':

    cf = CSGFoundry.Load()
    print(cf)

    f = Fold.Load(symbol="f")
    print(repr(f))

    st = stree(f)
    print(repr(st))

    eprint("np.abs(cf.inst-f.inst_f4).max()", globals(), locals() ) 
    w = epr("w = np.where( np.abs(cf.inst-f.inst_f4) > 0.0001 )",  globals(), locals() )

    check_inverse_pair(f, "inst", "iinst" )
    check_inverse_pair(f, "inst_f4", "iinst_f4" )
    compare_f_with_cf(f, cf ) 




