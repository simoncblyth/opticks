#!/usr/bin/env python
"""
G4OpBoundaryProcessTest_cf_QSimTest.py : random aligned comparison of boundary process
========================================================================================



"""


import os, numpy as np

A_FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest/propagate_at_boundary_s_polarized")
B_FOLD = "/tmp/G4OpBoundaryProcessTest"

if __name__ == '__main__':

     a_path = os.path.join(A_FOLD, "p.npy")
     b_path = os.path.join(B_FOLD, "p.npy")

     a = np.load(a_path)
     b = np.load(b_path)
     print("a.shape %10s : %s  " % (str(a.shape), a_path) )
     print("b.shape %10s : %s  " % (str(b.shape), b_path) )

     a_flag = a[:,3,3].view(np.uint32)  
     b_flag = b[:,3,3].view(np.uint32)  
     print( " a_flag %s " % str(np.unique(a_flag, return_counts=True)) ) 
     print( " b_flag %s " % str(np.unique(b_flag, return_counts=True)) ) 

     a_TransCoeff = a[:,1,3]
     b_TransCoeff = b[:,1,3]
     print( "a_TransCoeff %s " %  a_TransCoeff  ) 
     print( "b_TransCoeff %s " %  b_TransCoeff  ) 

     a_flat = a[:,0,3] 
     b_flat = b[:,0,3] 
     print(" a_flat %s " % a_flat)
     print(" b_flat %s " % b_flat)








