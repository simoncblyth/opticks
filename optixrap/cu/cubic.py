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



Polynomial long division, factoring out a root (x-x0) 

* every term other than remainder "c" gets a shadow with x0
* then as x0 is a root el(x0) is zero, so adding that to above, 
  regain original with the remainder.

::

    In [35]: el = expand((x-x0)*(x**2+(a+x0)*x+b+a*x0+x0**2))

    In [36]: el
    Out[36]: a*x**2 - a*x0**2 + b*x - b*x0 + x**3 - x0**3


    In [38]: ex
    Out[38]: a*x**2 + b*x + c + x**3

    In [39]: ex.subs(x, x0)
    Out[39]: a*x0**2 + b*x0 + c + x0**3   


::

    In [133]: cub.iroot = 1, 1j, 1j
    z**3 + z**2*(-1 - 2.0*I) + z*(-1.0 + 2.0*I) + 1.0
    complex coeff, descending 
    3 :     1     0 
    2 :    -1 -2.00000000000000 
    1 : -1.00000000000000 2.00000000000000 
    0 : 1.00000000000000     0 
    iroot: (1, 1j, 1j)  (from input) 
    oroot: [1.00000000000000, 1.0*I]  (from solving the expression) 

    In [134]: cub.iroot = 1, 1j, -1j   ## is that general ? conjugate roots to get real coeff ?
    z**3 - z**2 + 1.0*z - 1.0
    complex coeff, descending 
    3 : 1.00000000000000     0 
    2 : -1.00000000000000     0 
    1 : 1.00000000000000     0 
    0 : -1.00000000000000     0 
    iroot: (1, 1j, -1j)  (from input) 
    oroot: [1.00000000000000, -1.0*I, 1.0*I]  (from solving the expression) 

    In [135]: cub.iroot = 1, 2, 3
    z**3 - 6*z**2 + 11*z - 6
    complex coeff, descending 
    3 :     1     0 
    2 :    -6     0 
    1 :    11     0 
    0 :    -6     0 
    iroot: (1, 2, 3)  (from input) 
    oroot: [1, 2, 3]  (from solving the expression) 

    In [136]: cub.iroot = 3, 2+5j, 2-5j   ## https://en.wikipedia.org/wiki/Complex_conjugate_root_theorem
    z**3 - 7.0*z**2 + 41.0*z - 87.0
    complex coeff, descending 
    3 : 1.00000000000000     0 
    2 : -7.00000000000000     0 
    1 : 41.0000000000000     0 
    0 : -87.0000000000000     0 
    iroot: (3, (2+5j), (2-5j))  (from input) 
    oroot: [3.00000000000000, 2.0 - 5.0*I, 2.0 + 5.0*I]  (from solving the expression) 


"""

import math
from sympy import expand, symbols, simplify, collect, factor, solve, Pow, sqrt, Poly
from sympy import I, E, re, im
from coeff import get_coeff, get_coeff_, subs_coeff, print_coeff, expr_coeff

def cubic_0():
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

    c,b,a = cw3 
    isc = b**2 - 4*a*c    # quadratic, b^2 - 4ac ,  -b -sqrt() , -b+sqrt()

    r1 = (-b + sqrt(isc))/(2*a)
    r2 = (-b - sqrt(isc))/(2*a)

    print "r1:%s " % r1
    print "r2:%s " % r2    
    print "isc: %r " % isc

    rr = solve(ew3, w3)

def cubic_1():

    a,b,c = symbols("a b c")
    x,y = symbols("x y")
    ex = x**3 + a*x**2 + b*x + c
    print "ex:%s " % ex

    ey = expand(ex.subs([(x, y-a/3)]))
    print "ey:%s " % ey

    cy = get_coeff(ey, y, 4)

    eq = cy[0]
    ep = cy[1]

    print "eq:%s " % eq
    print "ep:%s " % ep

    z,p,q = symbols("z p q")


    simp = [
         (eq*27, q*27), 
         (ep*3, p*3) ]


    print_coeff(cy, "cy")

    subs_coeff(cy, simp)
    print_coeff(cy, "cy-simp")


    ez = expr_coeff(cy, z)
    print "ez:%s " % ez
 
 

class Cubic(object):
    """
    Expand complex cubic roots and collect coefficients
    """
    def __init__(self):
        pass

    poly = property(lambda self:Poly(self.expr, self.var))
    coeff = property(lambda self:self.poly.all_coeffs())

    gpoly = property(lambda self:Poly(self.gexpr, self.var))
    gcoeff = property(lambda self:self.gpoly.all_coeffs())


    def _set_iroot(self, args):

        assert len(args) == 3 
        _z0, _z1, _z2 = args

        self._z0 = _z0
        self._z1 = _z1
        self._z2 = _z2

        x0,x1,x2 = symbols("x0 x1 x2", real=True)
        y0,y1,y2 = symbols("y0 y1 y2", real=True)
        z0,z1,z2 = symbols("z0 z1 z2")

        z,x,y = symbols("z,x,y")

        ezz = collect(expand((z-z0)*(z-z1)*(z-z2)),z)
        self.gexpr = ezz 

        z0 = x0+I*y0
        z1 = x1+I*y1
        z2 = x2+I*y2

        ex = collect(expand((x-x0)*(x-x1)*(x-x2)),x)
        ez = collect(expand((z-z0)*(z-z1)*(z-z2)),z)

        xsub = [(x0, _z0.real),(x1,_z1.real),(x2,_z2.real)]
        ysub = [(y0, _z0.imag),(y1,_z1.imag),(y2,_z2.imag)]

        expr = ez.subs( xsub + ysub )
        var = z   
        oroot = solve(expr, var) 


        self.expr = expr
        self.var = var 
        self.oroot = oroot

        u,a,b,c = self.coeff
        assert u == 1

        dexpr = expand(expr.subs(z, y-a/3))
        self.dexpr = dexpr
        
        p = b - a * a/3.         
        q = c - a * b / 3. + 2. * a * a * a / 27. 

        p3 = p/3.
        q2 = q/2.

        p33 = p3**3
        q22 = q2**2

        delta = 4. * p * p * p + 27. * q * q 

        disc = delta/(27.*4)  
        sdisc = sqrt(math.fabs(disc))   

        if disc > 0:

            t0 = q2 + sdisc if q2 > 0. else -q2 + sdisc
            u0 = p33/t0
               
            t1 = -q/2. + sdisc   # non-robust quad
            u1 =  q/2. + sdisc 
        
            t,u = t0,u0
 
            ## when same signs of t,u ... precision will be terrible for small q 
            tcu = math.copysign(1., t) * math.pow(math.fabs(t),1./3) 
            ucu = math.copysign(1., u) * math.pow(math.fabs(u),1./3) 
            root0  = tcu - ucu - a/3
        else:
            t     = -q/2. 
            u     = sdisc           # sqrt of negated discrim :  sqrt( -[(p/3)**3 + (q/2)**2] )
            root0  = 2. * sqrt(-p/3.) * math.cos( math.atan2(u, t)/3.) 

            tcu = None
            ucu = None
        pass

        self.a = a 
        self.b = b
        self.c = c
  
        self.p = p 
        self.q = q
        self.p3 = p3
        self.q2 = q2
        self.p33 = p33
        self.q22 = q22 

        self.delta = delta
        self.disc = disc
        self.sdisc = sdisc
        self.root0 = root0
        self.t = t
        self.u = u
        self.tcu = tcu
        self.ucu = ucu
        self.tcu_ucu = tcu*ucu

        print self

    def _get_iroot(self):
        return self._z0, self._z1, self._z2 

    iroot = property(_get_iroot, _set_iroot)


    def __repr__(self):
        lines = []
        lines.append( repr(self.expr) )
        lines.append( "a:%s b:%s c:%s  " % (self.a,self.b,self.c) )
        lines.append( repr(self.dexpr) )
        lines.append( "p:%s q:%s (p/3)^3:%s  (q/2)^2: %s  " % (self.p,self.q,self.p33, self.q22) )
        lines.append( "delta:%s disc:%s sdisc:%s root0:%s " % (self.delta,self.disc,self.sdisc, self.root0) )
        lines.append( "t:%s u:%s " % (self.t,self.u))
        lines.append( "tcu:%s ucu:%s tcu_ucu:%s p3:%s  " % (self.tcu,self.ucu, self.tcu_ucu, self.p3 ))
        lines.append("complex coeff, descending ")
        for i, co in enumerate(self.coeff):
            lines.append("%s : %5s %5s " % (3-i, re(co), im(co) ))
        pass
        lines.append( "iroot: %s  (from input) " % repr(self.iroot) )
        lines.append( "oroot: %s  (from solving the expression) " % repr(self.oroot) )

        return "\n".join(lines)

if __name__ == '__main__':

   
    cub = Cubic()
    #cub.iroot = 1, 2, 3 
    cub.iroot = 3, 2+5j, 2-5j
    #cub.iroot = 1, 1, 1
  


