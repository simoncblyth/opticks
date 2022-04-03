#!/usr/bin/env python
"""
lambertian_direction_cf.py
===================================

* HMM : TOTALLY DIFFERENT : CANNOT BE RANDOM ALIGNED  
* YEP, THAT WAS PILOT ERROR : PRECOOKED USAGE WAS DISABLED IN THE SCRIPT

* DONE : added debug to check random consumption in qsim

2022-04-03 16:38:05.139 INFO  [4908425] [QSimTest<float>::main@614]  num_default 1000000 num 1000000 type 27 ni_tranche_size 100000 print_id -1
//QSim_quad_launch sim 0x703a40a00 quad 0x7042c0000 num_quad 1000000 dbg 0x703a40c00 type 27 name lambertian_direction 
//qsim.random_direction_marsaglia idx 0 u0     0.7402 u1     0.4385 
//qsim.lambertian_direction idx 0 count 1  u     0.5170 
//qsim.random_direction_marsaglia idx 0 u0     0.1570 u1     0.0714 
//qsim.random_direction_marsaglia idx 0 u0     0.4625 u1     0.2276 
//qsim.lambertian_direction idx 0 count 2  u     0.3294 

* DONE: use ./rng_sequence.sh to match those randoms with the precooked sequence for idx 0 

    In [1]: seq[0]
    Out[1]: 
    array([[0.74 , 0.438, 0.517, 0.157, 0.071, 0.463, 0.228, 0.329, 0.144, 0.188, 0.915, 0.54 , 0.975, 0.547, 0.653, 0.23 ],
           [0.339, 0.761, 0.546, 0.97 , 0.211, 0.947, 0.553, 0.978, 0.308, 0.18 , 0.387, 0.937, 0.691, 0.855, 0.489, 0.189],
           [0.507, 0.021, 0.958, 0.774, 0.418, 0.179, 0.259, 0.611, 0.9  , 0.446, 0.332, 0.73 , 0.976, 0.748, 0.488, 0.318],
           [0.712, 0.341, 0.468, 0.396, 0.001, 0.592, 0.87 , 0.632, 0.622, 0.555, 0.995, 0.525, 0.424, 0.138, 0.219, 0.791],

* TODO : do the same in bst : hmm will need to make local copy of the code in order to instrument it
* OOPS : no need  : found pilot error the precooked sequence was disabled 

::

    In [3]: np.where( np.abs( a-b) > 2e-5 )                                                                                                                                                                   
    Out[3]: 
    (array([542827, 563864, 821078, 894016, 924236, 924236, 924236]),
     array([1, 1, 0, 1, 0, 1, 2]))

    Out[6]: (array([542827, 924236, 924236, 924236]), array([1, 0, 1, 2]))
    In [7]: np.where( np.abs( a-b) > 1e-4 )
    Out[7]: (array([542827, 924236, 924236, 924236]), array([1, 0, 1, 2]))
    In [8]: np.where( np.abs( a-b) > 1e-3 )
    Out[8]: (array([924236, 924236, 924236]), array([0, 1, 2]))
    In [9]: np.where( np.abs( a-b) > 1e-4 )
    Out[9]: (array([542827, 924236, 924236, 924236]), array([1, 0, 1, 2]))

One direction in a million is totally off:

    In [10]: a[924236]
    Out[10]: array([ 0.172, -0.125,  0.977,  0.   ], dtype=float32)

    In [11]: b[924236]
    Out[11]: array([-0.44 , -0.746,  0.501,  0.   ], dtype=float32)

    In [12]: a[542827]
    Out[12]: array([-0.   , -0.001,  1.   ,  0.   ], dtype=float32)

    In [13]: b[542827]
    Out[13]: array([-0.   , -0.001,  1.   ,  0.   ], dtype=float32)


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


