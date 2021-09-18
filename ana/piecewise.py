#!/usr/bin/env python
"""
piecewise.py
==============

This succeeds to create a piecewise defined rindex using sympy 
and symbolically obtains Cerenkov s2 from that, 
using fixed BetaInverse = 1 

However, attempts to integrate that fail.

Hmm seems that sympy doesnt like a mix of symbolic and floating point, 
see cumtrapz.py for attempt to use scipy.integrate.cumtrapz to
check my s2 integrals.


https://stackoverflow.com/questions/43852159/wrong-result-when-integrating-piecewise-function-in-sympy


Programming for Computations, Hans Petter Langtangen
------------------------------------------------------

http://hplgit.github.io

http://hplgit.github.io/prog4comp/doc/pub/p4c-sphinx-Python/._pylight004.html

http://hplgit.github.io/prog4comp/doc/pub/p4c-sphinx-Python/index.html



Solvers seem not to work with sympy, so role simple bisection solver ?
-------------------------------------------------------------------------


https://personal.math.ubc.ca/~pwalls/math-python/roots-optimization/bisection/


"""
import logging
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt 

from sympy.plotting import plot
from sympy import Piecewise, piecewise_fold, Symbol, Interval, integrate, Max, Min
from sympy.abc import x, y
from sympy.solvers import solve

from opticks.ana.bisect import bisect

ri = np.array([
       [ 1.55 ,  1.478],
       [ 1.795,  1.48 ],
       [ 2.105,  1.484],
       [ 2.271,  1.486],
       [ 2.551,  1.492],
       [ 2.845,  1.496],
       [ 3.064,  1.499],
       [ 4.133,  1.526],
       [ 6.2  ,  1.619],
       [ 6.526,  1.618],
       [ 6.889,  1.527],
       [ 7.294,  1.554],
       [ 7.75 ,  1.793],
       [ 8.267,  1.783],
       [ 8.857,  1.664],
       [ 9.538,  1.554],
       [10.33 ,  1.454],
       [15.5  ,  1.454]
      ])


class PieceMaker(object):
    def __init__(self):
        pass

    def make_part(self, e, i, b=1.):
        e0, r0 = ri[i]
        e1, r1 = ri[i+1]

        v0 = ( 1 - b/r0 ) * ( 1 + b/r0 )
        v1 = ( 1 - b/r1 ) * ( 1 + b/r1 )

        fr = (e-e0)/(e1-e0)
        pt = ( v0*(1-fr) + v1*fr,  (e > e0) & (e < e1) ) 
        return pt 

    def make_piece(self, e, i, b=1.):
        """
        Max complicates the integrals, alternative is to manually control the range to keep s2 +ve 
        """
        e0, r0 = ri[i]
        e1, r1 = ri[i+1]

        v0 = ( 1 - b/r0 ) * ( 1 + b/r0 )
        v1 = ( 1 - b/r1 ) * ( 1 + b/r1 )

        fr = (e-e0)/(e1-e0)
        pt = ( Max(v0*(1-fr) + v1*fr,0),  (e > e0) & (e < e1) )    
        ot = (0, True )

        pw = Piecewise( pt, ot ) 
        return pw

    def make_parts(self, e):
        parts = []
        #parts.append( ( ri[0,1], e < ri[0,0] ) )
        for i in range(len(ri)-1):
            pt = self.make_part(e, i)
            parts.append(pt)
        pass
        #parts.append( ( ri[-1,1], e > ri[-1,0] ) )
        return parts    

 


  

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)    

    emn = ri[0,0]
    emx = ri[-1,0]
    emi = (emn+emx)/2.

    ck = np.zeros_like(ri)  
    ck[:,0] = ri[:,0]
    ck[:,1] = (1. - 1./ri[:,1])*(1.+1./ri[:,1])    

    e = Symbol('e', positive=True)

    pm = PieceMaker()

    # separate functions and integrals for each bin 
    s2 = {}
    is2 = {}
    for i in range( len(ri)-1 ):
        s2[i] = pm.make_piece( e, i, b=1.55 ) 
        is2[i] = integrate( s2[i], e )
        print("s2[%d]" % i, s2[i])
        print("is2[%d]" % i, is2[i])
    pass

    plot( *is2.values(), (e, emn, emx), show=True, adaptive=False, nb_of_points=500)   # cliff edges from each bin  

    is2a = sum(is2.values())    # dumb sum 

    #plot( is2a, (e, emn, emx), show=True )     

    is2a_norm = is2a/is2a.subs(e, emx)

    plot(is2a_norm, (e, emn, emx), show=True )

    #def curry(u): return lambda x:is2a_norm.subs(e, x) - u  
    #fn = curry(0.5)

    fn = lambda x:is2a_norm.subs(e, x) - 0.5
    #en = fsolve( fn, 9 )    

    #u = 0.5
    #en = solve( is2a_norm - u, e )    # just hangs

    en = bisect( fn, emn, emx, 20 )

    #en2 = nsolve( is2a_norm, e, emi )

    # scipy.optimize.newton( fn, 11 )   # fails

