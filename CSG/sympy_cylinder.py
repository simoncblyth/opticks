#!/usr/bin/env python
"""
sympy_cylinder.py
===================

Recast ray intersection with cylinder to requiring that the distance 
from cylinder axis line AB to a point C must be the cylinder radius.::


         [0,0,z2]  B +--r-+ C  (o + t v)
                     |   /
                     |  /
                     | /
                     |/
                   A +   [ 0,0,z1 ]



                   B + [0,0,z2]
                     |\    
                     | \
                     |r P  (o + t v) 
                     | /
                     |/
                   A + [ 0,0,z1 ]

* Area of triangle ABP with height r, |BA|d/2
* Area spanned by vectors u (BA) and v (BP), |u x v|/2
* equating areas gives: r = |BAxBP| / |BA| 

But the area can be given by another choice of side vectors too, so: 

* r = |APxBP|/|BA|


* AP = o+tv - [0,0,z1]
* BP = o+tv - [0,0,z2]
* |BA| = (z2-z1)    (z2 > z1 by definition)

* r*r = APxBP.dot(APxBP) / (z2-z1)**2

* fn(t) = APxBP.dot(APxBP) / (z2-z1)**2 - r*r


Comparing the a,b,c coeff from nmskTailOuterIITube with the old ones get factor of 20800.7383 in all three.
That is dz*dz::

    In [12]: (72.112*2)*(72.112*2)
    Out[12]: 20800.562175999996



Endcap intersects of the ray, p  = o + t v 
Plane eqn, n = [0,0,1]     p0 = [0,0,z1]  OR [0,0,z2] 

    (p - p0).n = 0    

This says that a vector in the plane which has no z component::

    oz + t vz  - z1 = 0  

    t = (z1 - oz)/vz 
    t = (z2 - oz)/vz 





"""

import numpy as np, sympy as sp
from collections import OrderedDict as odict 

def pp(q, q_label="?", note=""):

    tq = type(q)
    if tq is np.ndarray:
        q_type = "np"
    elif q.__class__.__name__  == 'MutableDenseMatrix':
        q_type = "sp.MutableDenseMatrix"
    else:
        q_type = "%s.%s" % ( tq.__module__ , tq.__name__ )
    pass

    sh = str(q.shape) if hasattr(q,"shape") else "" 

    print("\n%s : %s : %s : %s \n" % (q_label, q_type, sh, note) )

    if q_type.startswith("sp"):
        sp.pprint(q)
    elif q_type.startswith("np"):
        print(q)
    else:
        print(q)
    pass


#import sympy.physics.vector as spv

t = sp.symbols("t")
ox, oy, oz = sp.symbols("ox oy oz")
vx, vy, vz = sp.symbols("vx vy vz")
r,z1,z2 = sp.symbols("r z1 z2")

A = sp.Matrix([ [0], [0], [z1] ])
B = sp.Matrix([ [0], [0], [z2] ])
O = sp.Matrix([ [ox], [oy], [oz] ])  # column vector
V = sp.Matrix([ [vx], [vy], [vz] ])

P = O + t*V   # parametric ray eqn 

AP = P - A 
BP = P - B 
AB = B - A 

pp(P, "P")
pp(AP, "AP")
pp(BP, "BP")
pp(AB, "AB")

APBP = AP.cross(BP)
pp(APBP, "APBP", "AP.cross(BP) is perpendicular to axis, ie with zero z-component" )

apbp = sp.sqrt(APBP.dot(APBP)) 
pp(apbp, "apbp", "magnitude of the cross product, cross product proportional to ABC triangle area" )

ab = z2 - z1 

fn = (apbp*apbp)/(ab*ab) - r*r
#fn = apbp  - r*r*ab*ab

pp(fn, "fn")


collect_expand_fn = sp.collect( sp.expand(fn), t )  
pp(collect_expand_fn, "collect_expand_fn")

tfn = odict()
sfn = odict()

for i in range(3):
    tfn[i] = collect_expand_fn.coeff(t, i)
    sfn[i] = sp.simplify(tfn[i])
pass

for i in range(3): pp(tfn[i], "tfn[%d]"% i )
for i in range(3): pp(sfn[i], "sfn[%d]"% i )

