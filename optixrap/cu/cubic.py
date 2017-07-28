#!/usr/bin/env python
"""

Following along with http://mathworld.wolfram.com/CubicFormula.html


::

    In [73]: run cubic.py
    c
    c[0]:(2*p**3 - 9*p*q + 27*r)/27 
    c[1]:-(p**2 - 3*q)/3 
    c[2]:0 
    c[3]:1 



"""

from sympy import expand, symbols, simplify, collect, factor, solve, Pow, sqrt
from coeff import get_coeff, get_coeff_, subs_coeff, print_coeff, expr_coeff


if __name__ == '__main__':
    a2,a1,a0 = symbols("a2 a1 a0")
    x,y,z,l = symbols("x y z l")

    ez = z**3 + a2*z**2 + a1*z + a0   
    print "ez: %r (cubic with top coeff 1)" % ez

    ex = ez.subs(z, x-l ).subs(l, a2/3)   
    print "ex: %r (shift to kill **2 term)" % ex
    print "ex: %r (shift to kill **2 term)" % expand(ex)

    cx = get_coeff(ex, x, 4 )
    assert cx[2] == 0    # **2 term is killed

    p,q = symbols("p q", real=True)

    eq = (9*a2*a1 - 2*a2**3 - 27*a0)/27
    ep = ( 3*a1 - a2**2)/3

    simp = [
         (eq*27, q*27), 
         (ep*3, p*3) ]


    print_coeff(cx, "cx")

    subs_coeff(cx, simp)

    print_coeff(cx, "c-simp")

    ex2 = expr_coeff(cx, x)
    


    w= symbols("w")  
    ew = expand(ex2.subs(x, w-p/(3*w) ))    # Vieta's substitution

    ew2 = expand(ew*w**3)  # becomes quadratic in w**3

    w3= symbols("w3")  
    ew3 = ew2.subs(w**3, w3)

    cw3 = get_coeff(ew3,w3,3)

    a,b,c = cw3 
    isc = b**2 - 4*a*c    # quadratic, b^2 - 4ac ,  -b -sqrt() , -b+sqrt()

    r1 = (-b + sqrt(isc))/(2*a)
    r2 = (-b - sqrt(isc))/(2*a)



    


    print "isc: %r " % isc

    rr = solve(ew3, w3)


    delta,sq3,inv6sq3,ott = symbols("delta sq3 inv6sq3 ott")

    delta = 27*isc
    sq3 = sqrt(3)
    ott = Pow(3, -1)
    inv6sq3 = Pow(6*sq3, -1)




