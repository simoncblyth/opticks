#!/usr/bin/env python
"""
propagate_at_boundary_cf.py : random aligned comparison of boundary process
========================================================================================

Used via scripts which sets envvars::

   ./G4OpBoundaryProcessTest.sh cf 

"""

import os, numpy as np
#from opticks.ana.eprint import eprint

def eprint( expr, lprefix="", rprefix="", tail="" ):
    ret = eval(expr)
    lhs = "%s%s" % (lprefix, expr)
    rhs = "%s%s" % (rprefix, ret )
    print("%-50s : %s%s" % ( lhs, rhs, tail )   )   
    return ret 

def epr(arg, **kwa):
    p = arg.find("=")  
    if p > -1: 
        var_eq = arg[:p+1]
        expr = arg[p+1:]
        label = var_eq
    else:
        label, expr = "", arg 
    pass
    return eprint(expr, lprefix=label,  **kwa)


a_key = "A_FOLD"
b_key = "B_FOLD"

A_FOLD = os.environ[a_key] 
B_FOLD = os.environ[b_key] 


if __name__ == '__main__':
     print("a_key : %20s  A_FOLD : %s" % ( a_key, A_FOLD) )
     print("b_key : %20s  B_FOLD : %s" % ( b_key, B_FOLD) )

     a_path = os.path.join(A_FOLD, "p.npy")
     b_path = os.path.join(B_FOLD, "p.npy")

     aprd_path = os.path.join(A_FOLD, "prd.npy")
     bprd_path = os.path.join(B_FOLD, "prd.npy")

     a = np.load(a_path)
     b = np.load(b_path)
     print("a.shape %10s : %s  " % (str(a.shape), a_path) )
     print("b.shape %10s : %s  " % (str(b.shape), b_path) )

     aprd = np.load(aprd_path) if os.path.exists(aprd_path) else None
     bprd = np.load(bprd_path) if os.path.exists(bprd_path) else None
     if not aprd is None:
         print("aprd.shape %10s : %s  " % (str(aprd.shape), aprd_path) )
         eprint("aprd", lprefix="\n", rprefix="\n" )
     pass

     if not bprd is None:
         print("bprd.shape %10s : %s  " % (str(bprd.shape), bprd_path) )
         eprint("bprd", lprefix="\n", rprefix="\n" )
     pass


     a_flag  = epr("a_flag=a[:,3,3].view(np.uint32)")  
     b_flag  = epr("b_flag=b[:,3,3].view(np.uint32)")  
     w_flag  = epr("w_flag=np.where( a_flag != b_flag )")
     ua_flag = epr("ua_flag=np.unique(a_flag, return_counts=True)")
     ub_flag = epr("ub_flag=np.unique(b_flag, return_counts=True)")

     a_TransCoeff = epr("a_TransCoeff=a[:,1,3]")
     b_TransCoeff = epr("b_TransCoeff=b[:,1,3]")
     w_TransCoeff = epr("w_TransCoeff=np.where( np.abs( a_TransCoeff - b_TransCoeff) > 1e-6 )")

     a_flat = epr("a_flat=a[:,0,3]") 
     b_flat = epr("b_flat=b[:,0,3]") 
     w_flat = epr("w_flat=np.where(a_flat != b_flat)")

     expr_ = "w_ab%(i)s=np.where( np.abs(a[:,%(i)s,:3] - b[:,%(i)s,:3]) > 1e-6 )" 
     w_ab0 = epr(expr_ % dict(i=0) )
     w_ab1 = epr(expr_ % dict(i=1) )
     w_ab2 = epr(expr_ % dict(i=2) )
     w_ab3 = epr(expr_ % dict(i=3) )





