#!/usr/bin/env python
"""

Hmm maybe its impossible to get anywhere with integral without first fixing the BetaInverse


In [22]: pw.subs(b, 1.55)                                                                                                                                                                        
Out[22]: Piecewise((Max(0, 0.542862145685886*e - 3.9544951109741), (e > 7.294) & (e < 7.75)), (0, True))

In [23]: pw.subs(b, 1.)                                                                                                                                                                          
Out[23]: Piecewise((Max(0, 0.225957188630962*e - 1.06222481205998), (e > 7.294) & (e < 7.75)), (0, True))



"""
import numpy as np
import sympy as sym

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



if __name__ == '__main__':


     e, b = sym.symbols("e b")

     i = 11 

     e0, r0 = ri[i]
     e1, r1 = ri[i+1]

     em = (e0 + e1)/2.


     v0 = ( 1 - b/r0 ) * ( 1 + b/r0 )
     v1 = ( 1 - b/r1 ) * ( 1 + b/r1 )

     fr = (e-e0)/(e1-e0) 

     pt = ( sym.Max(v0*(1-fr) + v1*fr,0),  (e > e0) & (e < e1) )
     ot = (0, True )
     pw = sym.Piecewise( pt, ot )

     v = pw.subs(b, 1.55).subs(e, em)
     print(v)


