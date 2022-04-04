#!/usr/bin/env python
"""
reflect_diffuse_cf.py
===================================

Hmm direction is flipped ?::

    In [1]: a[0]                                                                                                                                                                                              
    Out[1]: 
    array([[  1.   ,   0.   ,   0.   ,   1.   ],
           [  0.244,  -0.067,   0.967,   1.   ],
           [  0.067,  -0.994,  -0.086, 500.   ],
           [  0.   ,   1.   ,   0.   , 500.   ]], dtype=float32)

    In [2]: b[0]                                                                                                                                                                                              
    Out[2]: 
    array([[  1.   ,   0.   ,   0.   ,   0.157],
           [ -0.244,   0.067,  -0.967,   0.   ],
           [ -0.067,  -0.996,  -0.052, 500.   ],
           [  0.   ,   1.   ,   0.   ,   0.   ]], dtype=float32)


After the flip, done with orient float in qsim, get match at 3-in-a-million level::

    In [3]: np.where( np.abs( a[:,1,:3] - b[:,1,:3] ) > 2e-5 )  # mom
    Out[3]: (array([780419, 780419, 800478, 947557]), array([0, 1, 1, 0]))
 
    In [4]: np.where( np.abs( a[:,2,:3] - b[:,2,:3] ) > 2e-5 )  # pol 
    Out[4]: (array([780419, 780419, 800478, 800478]), array([0, 2, 0, 2]))


"""
import os, numpy as np

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
NPY_NAME = os.environ["NPY_NAME"]


if __name__ == '__main__':
     print("a_key : %20s  A_FOLD : %s" % ( a_key, A_FOLD) )
     print("b_key : %20s  B_FOLD : %s" % ( b_key, B_FOLD) )
     a_path = os.path.join(A_FOLD, NPY_NAME)
     b_path = os.path.join(B_FOLD, NPY_NAME)

     a = np.load(a_path)
     b = np.load(b_path)
     print("a.shape %10s : %s  " % (str(a.shape), a_path) )
     print("b.shape %10s : %s  " % (str(b.shape), b_path) )


