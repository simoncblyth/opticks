#!/usr/bin/env python
"""
reflect_specular_cf.py
===================================

Specular does not depend on the random stream, so no surprise get perfect match.

::

    In [3]: np.where( np.abs( a[:,:,:3] - b[:,:,:3] ) > 2e-5 )
    Out[3]: (array([], dtype=int64), array([], dtype=int64), array([], dtype=int64))



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


