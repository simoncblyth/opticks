#!/usr/bin/env python
"""
G4OpBoundaryProcessTest_cf_QSimTest.py : random aligned comparison of boundary process
========================================================================================

Used via script which sets envvars::

   ./G4OpBoundaryProcessTest.sh cf 

"""

import os, numpy as np

a_key = "OPTICKS_QSIM_DSTDIR"
b_key = "OPTICKS_BST_DSTDIR"

A_FOLD = os.environ[a_key] 
B_FOLD = os.environ[b_key] 

def eprint( expr ):
    print("%s : %s" % ( expr, eval(expr) )   )


if __name__ == '__main__':
     print("a_key : %20s  A_FOLD : %s" % ( a_key, A_FOLD) )
     print("b_key : %20s  B_FOLD : %s" % ( b_key, B_FOLD) )

     a_path = os.path.join(A_FOLD, "p.npy")
     b_path = os.path.join(B_FOLD, "p.npy")

     a = np.load(a_path)
     b = np.load(b_path)
     print("a.shape %10s : %s  " % (str(a.shape), a_path) )
     print("b.shape %10s : %s  " % (str(b.shape), b_path) )

     a_flag = a[:,3,3].view(np.uint32)  
     b_flag = b[:,3,3].view(np.uint32)  
     print( "a_flag %s " % str(np.unique(a_flag, return_counts=True)) ) 
     print( "b_flag %s " % str(np.unique(b_flag, return_counts=True)) ) 
     eprint("np.where( a_flag != b_flag ) ")


     a_TransCoeff = a[:,1,3]
     b_TransCoeff = b[:,1,3]
     print( "a_TransCoeff %s " %  a_TransCoeff  ) 
     print( "b_TransCoeff %s " %  b_TransCoeff  ) 
     eprint("np.where( np.abs( a_TransCoeff - b_TransCoeff) > 1e-6 ) ")

     a_flat = a[:,0,3] 
     b_flat = b[:,0,3] 
     print("a_flat %s " % a_flat)
     print("b_flat %s " % b_flat)
     eprint( "np.where( a_flat != b_flat ) " )


     expr_ = "np.where( np.abs(a[:,%(i)s] - b[:,%(i)s]) > 1e-6 ) " 
     for i in range(4):
         eprint( expr_ % locals())
     pass






