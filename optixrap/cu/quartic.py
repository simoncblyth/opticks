#!/usr/bin/env python

"""


::

    In [166]: collect(expand(ex4), x )
    Out[166]: c0 - c1*l + c2*l**2 - c3*l**3 + c4*l**4 + c4*x**4 + x**3*(c3 - 4*c4*l) + x**2*(c2 - 3*c3*l + 6*c4*l**2) + x*(c1 - 2*c2*l + 3*c3*l**2 - 4*c4*l**3)


    In [168]: collect(expand(ex4), x )
    Out[168]: c0 - c1*c3/(4*c4) + c2*c3**2/(16*c4**2) - 3*c3**4/(256*c4**3) + c4*x**4 + x**2*(c2 - 3*c3**2/(8*c4)) + x*(c1 - c2*c3/(2*c4) + c3**3/(8*c4**2))

    In [169]: collect(expand(ex4), x ).coeff(x, 0)
    Out[169]: c0 - c1*c3/(4*c4) + c2*c3**2/(16*c4**2) - 3*c3**4/(256*c4**3)

    In [170]: collect(expand(ex4), x ).coeff(x, 1)
    Out[170]: c1 - c2*c3/(2*c4) + c3**3/(8*c4**2)

    In [171]: collect(expand(ex4), x ).coeff(x, 2)
    Out[171]: c2 - 3*c3**2/(8*c4)

    In [172]: collect(expand(ex4), x ).coeff(x, 3)
    Out[172]: 0

    In [173]: collect(expand(ex4), x ).coeff(x, 4)
    Out[173]: c4



"""

from sympy import expand, symbols, simplify, collect, factor, solve, Pow
from coeff import get_coeff, get_coeff_, subs_coeff, print_coeff, expr_coeff


if __name__ == '__main__':
    a0,a1,a2,a3,a4 = symbols("a0 a1 a2 a3 a4")
    q0,q1,q2,q3,q4 = symbols("q0 q1 q2 q3 q4")

    x,z,l = symbols("x z l")

    ex_ = Pow(x,4) + a3*Pow(x,3) + a2*Pow(x,2) + a1*Pow(x,1) + a0   # setting a4 to 1 

    ex = ex_.subs(x, x-l ).subs(l, a3/4)   # picking this l kills the x**3 term 


    c = get_coeff(ex, x, 5 )

    assert c[3] == 0    # x**3 term is killed


    p,q,r = symbols("p q r")

    simp = [
            ( a2 - 3*a3**2/8 , p ),
            ( a1 - a2*a3/2 + a3**3/8 , q),
            ( -a1*a3/4 + a2*a3**2/16 - 3*a3**4/256, r )
            ]

    

    cs = get_coeff_(ex, x, 5)
    print_coeff(cs, "cs")

    #subs_coeff(cs, simp)

    #print_coeff(cs, "cs")

    #ex2 = expr_coeff(cs, x)






