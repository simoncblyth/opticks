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

"""
import logging
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt 

from sympy.plotting import plot
from sympy import Piecewise, log, piecewise_fold, Symbol, Interval, integrate
from sympy.abc import x, y

if __name__ == '__main__':

    e = Symbol('e', positive=True)
    b = Symbol('b', positive=True)

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


    parts = []
    parts.append( ( ri[0,1], e < ri[0,0] ) )
    for i in range(len(ri)-1):
        e0, v0 = ri[i]
        e1, v1 = ri[i+1]
        fr = (e-e0)/(e1-e0)
        #parts.append( ( v1, (e > e0) & (e < e1) )  )
        parts.append( ( v0*(1-fr) + v1*fr,  (e > e0) & (e < e1) )  )
    pass
    parts.append( ( ri[-1,1], e > ri[-1,0] ) )
      

    log.info("Piecewise... p ")
    p = Piecewise( *parts )

    log.info("p.subs... ")
    log.info(p.subs(e,1))
    log.info(p.subs(e,5))

    log.info("s2...")
    s2 = ( 1 - 1/p )*( 1 + 1/p )
    s2s = s2.simplify()

    #log.info("s2_1...")
    #s2_1 = s2.subs(b, 1)

    log.info("integrate... s2_1i ")
    s2i = integrate( s2, e )

    log.info("plot...")
    p1 = plot( s2i, (e, 0, 16), show=False )

    log.info("show...")
    p1.show()

    log.info("done...")
    s2si = integrate( s2s, (e, 1.55, 15.5) )



