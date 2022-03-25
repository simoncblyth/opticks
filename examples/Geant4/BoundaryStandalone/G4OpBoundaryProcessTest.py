#!/usr/bin/env python
"""
G4OpBoundaryProcessTest.py : Standalone testing of boundary process
=====================================================================

Normal incidence special case between media with rindex 1.0 and 1.5.
Checking how often get FresnelReflection and FresnelRefraction at normal incidence for 100k::

    (array([2, 3], dtype=uint32), array([96020,  3980]))

Status flags::

    2 FresnelRefraction 
    3 FresnelReflection 
 
https://en.wikipedia.org/wiki/Fresnel_equations

That corresponds to expectation for normal incidence Reflectance::

            |  n1 - n2 |2               
        R = | -------- |    =  (0.5/2.5)^2 
            |  n1 + n2 |            

    In [2]: (0.5/2.5)*(0.5/2.5)
    Out[2]: 0.04000000000000001

Checking the last random in FresnelReflection and FresnelRefraction 
sub-samplescan see that TransCoeff must be very close to 0.96::

    In [10]: flat[flag==3].min()  # 3:FresnelReflection
    Out[10]: 0.96001875

    In [11]: flat[flag==2].max()  # 2:FresnelRefraction
    Out[11]: 0.9599877
    
"""
import os, numpy as np

def eprint( expr, lprefix="", rprefix="" ):
    lhs = "%s%s" % (lprefix, expr)
    rhs = "%s%s" % (rprefix, eval(expr) )
    print("%s : %s" % ( lhs, rhs )   )   
pass

if __name__ == '__main__':

    key = "OPTICKS_BST_DSTDIR"
    FOLD = os.environ[key] 
    print("key : %20s  FOLD : %s" % (key, FOLD) )

    p_path = os.path.join(FOLD, "p.npy")

    p = np.load(p_path)
    print("p.shape %10s : %s  " % (str(p.shape), p_path) )

    flag = p[:,3,3].view(np.uint32)  
    eprint( "np.unique(flag, return_counts=True)" ) 

    TransCoeff = p[:,1,3]
    print( "TransCoeff %s " %  TransCoeff  ) 

    flat = p[:,0,3] 
    print(" flat %s " % flat)
    #print( flat[flag==3].min() )  
    #print( flat[flag==2].max() )
    



