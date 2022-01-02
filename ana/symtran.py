#!/usr/bin/env python
"""
symtran.py
============


In [22]: T                                                                                                                                                                                                
Out[22]: 
Matrix([
[ 1,  0,  0, 0],
[ 0,  1,  0, 0],
[ 0,  0,  1, 0],
[tx, ty, tz, 1]])

In [23]: R                                                                                                                                                                                                
Out[23]: 
Matrix([
[sin(theta)*cos(phi), sin(phi)*sin(theta),  cos(theta), 0],
[cos(phi)*cos(theta), sin(phi)*cos(theta), -sin(theta), 0],
[          -sin(phi),            cos(phi),           0, 0],
[                  0,                   0,           0, 1]])

In [24]: T*R                                                                                                                                                                                              
Out[24]: 
Matrix([
[                                          sin(theta)*cos(phi),                                           sin(phi)*sin(theta),                    cos(theta), 0],
[                                          cos(phi)*cos(theta),                                           sin(phi)*cos(theta),                   -sin(theta), 0],
[                                                    -sin(phi),                                                      cos(phi),                             0, 0],
[tx*sin(theta)*cos(phi) + ty*cos(phi)*cos(theta) - tz*sin(phi), tx*sin(phi)*sin(theta) + ty*sin(phi)*cos(theta) + tz*cos(phi), tx*cos(theta) - ty*sin(theta), 1]])

In [25]: R*T                                                                                                                                                                                              
Out[25]: 
Matrix([
[sin(theta)*cos(phi), sin(phi)*sin(theta),  cos(theta), 0],
[cos(phi)*cos(theta), sin(phi)*cos(theta), -sin(theta), 0],
[          -sin(phi),            cos(phi),           0, 0],
[                 tx,                  ty,          tz, 1]])

"""
from collections import OrderedDict as odict 
import numpy as np
import sympy as sp

np_ = lambda _:np.array(_).astype(np.float64)



def pp(q, q_label="?", note=""):

    if type(q) is np.ndarray:
        q_type = "np"
    elif q.__class__.__name__  == 'MutableDenseMatrix':
        q_type = "sp.MutableDenseMatrix"
    else:
        q_type = "?"
    pass
    print("\n%s : %s : %s : %s \n" % (q_label, q_type, str(q.shape), note) )

    if q_type.startswith("sp"):
        sp.pprint(q)
    elif q_type.startswith("np"):
        print(q)
    else:
        print(q)
    pass



radius, theta, phi = sp.symbols("radius theta phi")                                                                                                                                               
tx, ty, tz = sp.symbols("tx ty tz")
x, y, z, w = sp.symbols("x y z w")

R = sp.Matrix([
       [ sp.sin(theta)*sp.cos(phi) , sp.sin(theta)*sp.sin(phi) , sp.cos(theta)  ,  0 ],
       [ sp.cos(theta)*sp.cos(phi) , sp.cos(theta)*sp.sin(phi) , -sp.sin(theta) ,  0 ],
       [ -sp.sin(phi)              , sp.cos(phi)               , 0              ,  0 ],
       [ 0                         , 0                         , 0              ,  1 ]
       ])


T = sp.Matrix([ [1,0,0,0], [0,1,0,0], [0,0,1,0], [tx, ty, tz, 1] ])

V = sp.Matrix([[x, y, z, w]] )        # row vector (same as with single bracket) 

P = sp.Matrix([[x], [y], [z], [w] ])  # column vector


pp(R,"R")
pp(T,"T")
pp(V,"V")
pp(P,"P")
pp(V*T,"V*T")
pp(T.T*P, "T.T*P")
pp(T*R, "T*R")  
pp(R*T,"R*T", "simple tx ty yz 1 last row" )  



t = 0.25
p = 0.0
loc = odict()
loc["midlat"] = [(theta,t*np.pi),(phi,p*np.pi)] 

pos = odict()
pos["origin"] = [(x,0), (y,0), (z,0), (w,1)] 

tlate = [(tx, 100), (ty, 200), (tz, 300) ]

midlat = loc["midlat"]
origin = pos["origin"]


pp(R.subs(midlat), "R.subs(midlat)")
pp(T.subs(tlate),  "T.subs(tlate)")
pp(P.subs(origin), "P.subs(origin)")
pp(V.subs(origin), "V.subs(origin)")

pp(T.subs(tlate) * P.subs(origin), "T.subs(tlate) * P.subs(origin)", "no translation like this" )  
pp(T.subs(tlate).T * P.subs(origin), "T.subs(tlate).T * P.subs(origin)", "have to transpose the T to get the translation" )  

pp(V.subs(origin) * T.subs(tlate) , "V.subs(origin) * T.subs(tlate)", "when using row vector V have to mutltiply to the right with the un-transposed T to get translation " )


nV = np_(V.subs(origin))
nT = np_(T.subs(tlate))

pp(nV, "nV = np_(V.subs(origin)) ")
pp(nV.T, "nV.T ")

pp(nT, "nT = np_(T.subs(tlate)) ")

pp( np.dot(nV, nT), "np.dot(nV, nT)" )

pp( np.dot(nT, nV.T), "np.dot(nT, nV.T)" )


