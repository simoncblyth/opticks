#!/usr/bin/env python
"""

::

   (x*x + y*y + z*z + R*R + r*r)^2 = 4R*R(x*x + y*y)  

::

    In [18]: (x*x + y*y + z*z + R*R - r*r)**2 - 4*R*R*(x*x + y*y)
    Out[18]: -4*R**2*(x**2 + y**2) + (R**2 - r**2 + x**2 + y**2 + z**2)**2


    In [29]: expr2
    Out[29]: -4*R**2*((ox + sx*t)**2 + (oy + sy*t)**2) + (R**2 - r**2 + (ox + sx*t)**2 + (oy + sy*t)**2 + (oz + sz*t)**2)**2


    In [35]: expand(expr2)
    Out[35]: R**4 - 2*R**2*ox**2 - 4*R**2*ox*sx*t - 2*R**2*oy**2 - 4*R**2*oy*sy*t + 2*R**2*oz**2 + 4*R**2*oz*sz*t - 2*R**2*r**2 - 2*R**2*sx**2*t**2 - 2*R**2*sy**2*t**2 + 2*R**2*sz**2*t**2 + ox**4 + 4*ox**3*sx*t + 2*ox**2*oy**2 + 4*ox**2*oy*sy*t + 2*ox**2*oz**2 + 4*ox**2*oz*sz*t - 2*ox**2*r**2 + 6*ox**2*sx**2*t**2 + 2*ox**2*sy**2*t**2 + 2*ox**2*sz**2*t**2 + 4*ox*oy**2*sx*t + 8*ox*oy*sx*sy*t**2 + 4*ox*oz**2*sx*t + 8*ox*oz*sx*sz*t**2 - 4*ox*r**2*sx*t + 4*ox*sx**3*t**3 + 4*ox*sx*sy**2*t**3 + 4*ox*sx*sz**2*t**3 + oy**4 + 4*oy**3*sy*t + 2*oy**2*oz**2 + 4*oy**2*oz*sz*t - 2*oy**2*r**2 + 2*oy**2*sx**2*t**2 + 6*oy**2*sy**2*t**2 + 2*oy**2*sz**2*t**2 + 4*oy*oz**2*sy*t + 8*oy*oz*sy*sz*t**2 - 4*oy*r**2*sy*t + 4*oy*sx**2*sy*t**3 + 4*oy*sy**3*t**3 + 4*oy*sy*sz**2*t**3 + oz**4 + 4*oz**3*sz*t - 2*oz**2*r**2 + 2*oz**2*sx**2*t**2 + 2*oz**2*sy**2*t**2 + 6*oz**2*sz**2*t**2 - 4*oz*r**2*sz*t + 4*oz*sx**2*sz*t**3 + 4*oz*sy**2*sz*t**3 + 4*oz*sz**3*t**3 + r**4 - 2*r**2*sx**2*t**2 - 2*r**2*sy**2*t**2 - 2*r**2*sz**2*t**2 + sx**4*t**4 + 2*sx**2*sy**2*t**4 + 2*sx**2*sz**2*t**4 + sy**4*t**4 + 2*sy**2*sz**2*t**4 + sz**4*t**4

    In [58]: factor(c4)
    Out[58]: (sx**2 + sy**2 + sz**2)**2

    In [59]: factor(c3)
    Out[59]: 4*(sx**2 + sy**2 + sz**2)*(ox*sx + oy*sy + oz*sz)

    In [60]: factor(c2)
    Out[60]: -2*(R**2*sx**2 + R**2*sy**2 - R**2*sz**2 - 3*ox**2*sx**2 - ox**2*sy**2 - ox**2*sz**2 - 4*ox*oy*sx*sy - 4*ox*oz*sx*sz - oy**2*sx**2 - 3*oy**2*sy**2 - oy**2*sz**2 - 4*oy*oz*sy*sz - oz**2*sx**2 - oz**2*sy**2 - 3*oz**2*sz**2 + r**2*sx**2 + r**2*sy**2 + r**2*sz**2)

    In [61]: factor(c1)
    Out[61]: -4*(R**2*ox*sx + R**2*oy*sy - R**2*oz*sz - ox**3*sx - ox**2*oy*sy - ox**2*oz*sz - ox*oy**2*sx - ox*oz**2*sx + ox*r**2*sx - oy**3*sy - oy**2*oz*sz - oy*oz**2*sy + oy*r**2*sy - oz**3*sz + oz*r**2*sz)

    In [62]: factor(c0)
    Out[62]: R**4 - 2*R**2*ox**2 - 2*R**2*oy**2 + 2*R**2*oz**2 - 2*R**2*r**2 + ox**4 + 2*ox**2*oy**2 + 2*ox**2*oz**2 - 2*ox**2*r**2 + oy**4 + 2*oy**2*oz**2 - 2*oy**2*r**2 + oz**4 - 2*oz**2*r**2 + r**4


    In [64]: solve(expr2, t )
    Out[64]: 
    [Piecewise((-sqrt(-2*(-((-2*R**2*sx**2 - 2*R**2*sy**2 + 2*R**2*sz**2 + 6*ox**2*sx**2 + 2*ox**2*sy**2 + 2*ox**2*sz**2 + 8*ox*oy*sx*sy + 8*ox*oz*sx*sz + 2*oy**2*sx**2 + 6*oy**2*sy**2 + 2*oy**2*sz**2 + 8*oy*oz*sy*sz + 2*oz**2*sx**2 + 2*oz**2*sy**2 + 6*oz**2*sz**2 - 2*r**2*sx**2 - 2*r**2*sy**2 - 2*r**2*sz**2)/(sx**4 + 2*sx**2*sy**2 + 2*sx**2*sz**2 + sy**4 + 2*sy**2*sz**2 + sz**4) - 3*(4*ox*sx + 4*oy*sy + 4*oz*sz)**2/(8*(sx**2 + sy**2 + sz**2)**2))**3/108 + ((-2*R**2*sx**2 - 2*R**2*sy**2 + 2*R**2*sz**2 + 6*ox**2*sx**2 + 2*ox**2*sy**2 + 2*ox**2*sz**2 + 8*ox*oy*sx*sy + 8*ox*oz*sx*sz + 2*oy**2*sx**2 + 6*oy**2*sy**2 + 2*oy**2*sz**2 + 8*oy*oz*sy*sz + 2*oz**2*sx**2 + 2*oz**2*sy**2 + 6*oz**2*sz**2 - 2*r**2*sx**2 - 2*r**2*sy**2 - 2*r**2*sz**2)/(sx**4 + 2*sx**2*sy**2 + 2*sx**2*sz**2 + sy**4 + 2*sy**2*sz**2 + sz**4) - 3*(4*ox*sx + 4*oy*sy + 4*oz*sz)**2/(8*(sx*
...


    In [66]: s = _64

    In [68]: type(s)
    Out[68]: list

    In [69]: len(s)
    Out[69]: 4

    In [71]: type(s[0])
    Out[71]: sympy.functions.elementary.piecewise.Piecewise

    In [72]: type(s[1])
    Out[72]: sympy.functions.elementary.piecewise.Piecewise

    In [73]: type(s[2])
    Out[73]: sympy.functions.elementary.piecewise.Piecewise

    In [74]: type(s[3])
    Out[74]: sympy.functions.elementary.piecewise.Piecewise



    In [56]: ex.subs([(ox,0),(oy,0),(oz,0),(sx,1),(sy,0),(sz,0)])
    Out[56]: -4*R**2*t**2 + (R**2 - r**2 + t**2)**2

    In [57]: factor(ex.subs([(ox,0),(oy,0),(oz,0),(sx,1),(sy,0),(sz,0)]))
    Out[57]: (-R + r - t)*(-R + r + t)*(R + r - t)*(R + r + t)

    In [60]: solve(ex.subs([(ox,0),(oy,0),(oz,0),(sx,1),(sy,0),(sz,0)]), t )
    Out[60]: [-R - r, -R + r, R - r, R + r]

    In [62]: solve(ex.subs([(ox,0),(oy,0),(oz,0),(sx,0),(sy,0),(sz,1)]), t )
    Out[62]: [-sqrt(-R**2 + r**2), sqrt(-R**2 + r**2)]



"""

from sympy import symbols, Pow
from coeff import get_coeff, subs_coeff, print_coeff, expr_coeff

def torus_0():

    x,y,z,r,R = symbols("x y z r R")

    expr = (x*x + y*y + z*z + R*R - r*r)**2 - 4*R*R*(x*x + y*y)

    ox,oy,oz,t,sx,sy,sz,SS,OO,OS = symbols("ox oy oz t sx sy sz SS OO OS")

    ray = [(x,ox+t*sx),(y,oy+t*sy),(z,oz+t*sz)]
    simp = [(sx**2+sy**2+sz**2,SS),(ox**2+oy**2+oz**2,OO),(ox*sx+oy*sy+oz*sz,OS)]

    expr2 = expr.subs(ray+simp)

    x_expr2 = expand(expr2)
    t_expr2 = collect(x_expr2, t )

    c = get_coeff( expr2, t, 5 )

    subs_coeff(c, simp)
    print_coeff(c) 

    return expr2, c



#def torus_1():
if 1:
    """
    http://www.cosinekitty.com/raytrace/chapter13_torus.html

    Symbolic manipulations matching cosinekitty example

    * simplifying coeffs separately then putting 
      back together seems an effective approach

    """
    x,y,z,A,B = symbols("x y z A B")

    _sq_lhs = (x*x + y*y + z*z + A*A - B*B) 
    _rhs = 4*A*A*(x*x + y*y)
   
    ox,oy,oz,t,sx,sy,sz,SS,OO,OS = symbols("ox oy oz t sx sy sz SS OO OS")

    ray = [(x,ox+t*sx),(y,oy+t*sy),(z,oz+t*sz)]
    simp0 = [(sx**2+sy**2+sz**2,SS),(ox**2+oy**2+oz**2,OO),(ox*sx+oy*sy+oz*sz,OS)]


    G,H,I,J,K,L = symbols("G H I J K L")
  
    simp1 = [
        (4*A*A*(sx*sx+sy*sy),   G),
        (8*A*A*(ox*sx+oy*sy),   H),
        (4*A*A*(ox*ox+oy*oy),   I),
        (sx*sx+sy*sy+sz*sz,     J),
        (ox*sx+oy*sy+oz*sz,     K/2),   # fails to sub with 2 on other side
        (ox*ox+oy*oy+oz*oz+A*A-B*B, L)] 

    sq_lhs = _sq_lhs.subs(ray + simp0)
    rhs = _rhs.subs(ray + simp0)


    cl = get_coeff(sq_lhs, t, 3)
    cr = get_coeff(rhs, t, 3)

    subs_coeff(cl, simp1)
    subs_coeff(cr, simp1)

    print_coeff(cl, "cl")
    print_coeff(cr, "cr")

    SQ_LHS = expr_coeff(cl, t)
    RHS = expr_coeff(cr, t)

    ex = Pow(SQ_LHS, 2) - RHS

    c = get_coeff(ex, t, 5)

    print_coeff(c, "c")

    #return ex, cl


if __name__ == '__main__':
    pass
    #ex, c = torus_0()
    #ex, c = torus_1()


