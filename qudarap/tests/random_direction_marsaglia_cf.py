#!/usr/bin/env python
"""
random_direction_marsaglia_cf.py
===================================

* clearly 1e-6 is too stringent a criteria, that falls foul of too many float/double differences 
  in the x and y with none in z : that makes sense as only x and y suffer the sqrtf 

* relaxing to 2e-5 gets down to 1-in-million differences
* relaxing to 1e-5 gets down to 5-in-million differences (actually sort of 3M as 3 components of direction)

* I was expecting to get a few which go around the marsaglia rejection sampling  while loop 
  different numbers of times resulting in totally different values (due to cut edgers) 
  but there are no entries with totally different values at the 1 in a million level 

* so differences at this level can all be ascribed to float/double difference.  

::

    In [8]: np.where( np.abs( a - b ) > 1e-6 )[0]
    Out[8]: array([   386,    386,   1536, ..., 998618, 998618, 999787])

    In [9]: np.where( np.abs( a - b ) > 1e-6 )[0].shape
    Out[9]: (1017,)

    In [10]: np.where( np.abs( a - b ) > 1e-6 )[1].shape
    Out[10]: (1017,)

    In [11]: np.where( np.abs( a - b ) > 1e-6 )[1]
    Out[11]: array([0, 1, 1, ..., 0, 1, 0])

    In [12]: np.unique( np.where( np.abs( a - b ) > 1e-6 )[1], return_counts=True )
    Out[12]: (array([0, 1]), array([454, 563]))

    In [13]: np.where( np.abs( a - b ) > 1e-5 )[0]
    Out[13]: array([104838, 197893, 237931, 676309, 894016, 894016, 950910])

          ## relaxing criteria to 1e-5 gets to 5 in a million level 

    In [25]: np.where( np.abs( a - b ) > 2e-5 )
    Out[25]: (array([894016]), array([1]))

    In [14]: np.where( np.abs( a - b ) > 1e-4 )[0]
    Out[14]: array([], dtype=int64)

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


