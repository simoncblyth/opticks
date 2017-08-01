#!/usr/bin/env python

"""
::

    In [194]: ex
    Out[194]: -3*a**4/256 + a**3*z/8 + a**2*b/16 - 3*a**2*z**2/8 - a*b*z/2 - a*c/4 + b*z**2 + c*z + d + z**4

    In [195]: ex.coeff(z, 0)
    Out[195]: -3*a**4/256 + a**2*b/16 - a*c/4 + d    ## g

    In [196]: ex.coeff(z, 1)     
    Out[196]: a**3/8 - a*b/2 + c      ## f 

    In [197]: ex.coeff(z, 2)    
    Out[197]: -3*a**2/8 + b       ## e

    In [198]: ex.coeff(z, 3)
    Out[198]: 0

    In [199]: ex.coeff(z, 4)
    Out[199]: 1

"""

from sympy import expand, symbols, simplify, collect, factor, solve, Pow, Poly
from sympy import I, E, re, im
from coeff import get_coeff, get_coeff_, subs_coeff, print_coeff, expr_coeff


def quartic_0():
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



def quartic_1():
 
    x,z,l = symbols("x,z,l")
    a,b,c,d= symbols("a,b,c,d")

    ex = x**4 + a*x**3 + b*x**2 + c*x + d   
    ex = expand(ex.subs(x, z-a/4 ))   # shift kills the x**3 term 

    e,f,g = symbols("e,f,g")

    #  x^4 + a*x^3 + b*x^2 + c*x + d = 0
    #  z^4 +   0   + e z^2 + f z + g = 0 

    e = ex.coeff(z, 2)    # -3*a**2/8 + b 
    f = ex.coeff(z, 1)    # a**3/8 - a*b/2 + c
    g = ex.coeff(z, 0)    # -3*a**4/256 + a**2*b/16 - a*c/4 + d 

    # f = 0  -> degenerates to quadratic in z^2
    #  z^4 + e z^2 + g = 0 


"""

In [10]: ex = 4*(u-e)*(u**2/4 - g)

In [11]: eu = collect(expand(ex), u) ; eu 
Out[11]: 4*e*g - e*u**2 - 4*g*u + u**3

In [15]: Poly(eu,u).all_coeffs()
Out[15]: [1, -e, -4*g, 4*e*g]

"""


class Quartic(object):
    def __init__(self):
        pass

    poly = property(lambda self:Poly(self.expr, self.var))
    coeff = property(lambda self:self.poly.all_coeffs())

    gpoly = property(lambda self:Poly(self.gexpr, self.var))
    gcoeff = property(lambda self:self.gpoly.all_coeffs())

    def _set_iroot(self, args):

        assert len(args) == 4
        _z0, _z1, _z2, _z3 = args

        self._z0 = _z0
        self._z1 = _z1
        self._z2 = _z2
        self._z3 = _z3

        x0,x1,x2,x3 = symbols("x0 x1 x2 x3", real=True)
        y0,y1,y2,y3 = symbols("y0 y1 y2 y3", real=True)
        z0,z1,z2,z3 = symbols("z0 z1 z2 z3")

        z,x,y = symbols("z,x,y")

        ezz = collect(expand((z-z0)*(z-z1)*(z-z2)*(z-z3)),z)
        self.gexpr = ezz 

        z0 = x0+I*y0
        z1 = x1+I*y1
        z2 = x2+I*y2
        z3 = x3+I*y3

        ez = collect(expand((z-z0)*(z-z1)*(z-z2)*(z-z3)),z)

        xsub = [(x0, _z0.real),(x1,_z1.real),(x2,_z2.real),(x3,_z3.real)]
        ysub = [(y0, _z0.imag),(y1,_z1.imag),(y2,_z2.imag),(y3,_z3.imag)]

        expr = ez.subs( xsub + ysub )
        var = z   
        oroot = solve(expr, var) 

        self.expr = expr
        self.var = var 
        self.oroot = oroot

        allco = self.coeff
        print "allco: %s " % repr(allco)
        u,a,b,c,d = allco
        assert u == 1
        self.a = a 
        self.b = b
        self.c = c
        self.d = d

        dexpr = expand(expr.subs(z, y-a/4))
        self.dexpr = dexpr

        print self


    def _get_iroot(self):
        return self._z0, self._z1, self._z2, self._z3

    iroot = property(_get_iroot, _set_iroot)


    def __repr__(self):
        lines = []
        lines.append( repr(self.expr) )
        lines.append( "a:%s b:%s c:%s d:%s  " % (self.a,self.b,self.c,self.d) )
        lines.append( repr(self.dexpr) )
        lines.append("complex coeff, descending ")
        for i, co in enumerate(self.coeff):
            lines.append("%s : %5s %5s " % (3-i, re(co), im(co) ))
        pass
        lines.append( "iroot: %s  (from input) " % repr(self.iroot) )
        lines.append( "oroot: %s  (from solving the expression) " % repr(self.oroot) )

        return "\n".join(lines)







def neumark():
    """


    p12::

        In [25]: ef
        Out[25]: (A*x**2 + G*x + H)*(A*x**2 + g*x + h)

        In [26]: expand(ef)
        Out[26]: A**2*x**4 + A*G*x**3 + A*H*x**2 + A*g*x**3 + A*h*x**2 + G*g*x**2 + G*h*x + H*g*x + H*h

        In [27]: expand(ef/A)
        Out[27]: A*x**4 + G*x**3 + H*x**2 + g*x**3 + h*x**2 + G*g*x**2/A + G*h*x/A + H*g*x/A + H*h/A

        In [28]: Poly(expand(ef/A),x).all_coeffs()
        Out[28]: [A, G + g, (A*H + A*h + G*g)/A, (G*h + H*g)/A, H*h/A]

        In [30]: Poly(ex,x).all_coeffs()
        Out[30]: [A, B, C, D, E]

    """
    pass



def cardano():
    """


    cy coeffs of depressed quartic:: 

         [1, 
          0, 
         -3*a**2/8 + b, 
          a**3/8 - a*b/2 + c, 
          -3*a**4/256 + a**2*b/16 - a*c/4 + d]

        In [53]: [2*e, e*e - 4* g, -f*f ]
        Out[53]: 
        [-3*a**2/4 + 2*b,
         3*a**4/64 - a**2*b/4 + a*c - 4*d + (-3*a**2/8 + b)**2,
         (-a**3/8 + a*b/2 - c)*(a**3/8 - a*b/2 + c)]

        In [54]: map(expand, [2*e, e*e - 4* g, -f*f ])
        Out[54]: 
        [-3*a**2/4 + 2*b,
         3*a**4/16 - a**2*b + a*c + b**2 - 4*d,
         -a**6/64 + a**4*b/8 - a**3*c/4 - a**2*b**2/4 + a*b*c - c**2]

    """
    pass
    a,b,c,d,x,y,z = symbols("a:d,x:z")
    ex = x**4 + a*x**3 + b*x**2 + c*x + d
    ey = expand(ex.subs(x,y-a/4))
    cy = Poly(ey,y).all_coeffs()   
    e,f,g = cy[2:] 




def neumark():
    A,B,C,D,E = symbols("A:E")
    x = symbols("x")
    ex = A*Pow(x,4)+B*Pow(x,3)+C*Pow(x,2)+D*Pow(x,1)+E*Pow(x,0)


    G,g,H,h = symbols("G,g,H,h")
    ef = (A*Pow(x,2)+G*Pow(x,1)+H)*(A*Pow(x,2)+g*Pow(x,1)+h)
 

def ferrari():
    """

    In [2]: efac
    Out[2]: (G*x + H + x**2)*(g*x + h + x**2)

    In [3]: expand(efac)
    Out[3]: G*g*x**2 + G*h*x + G*x**3 + H*g*x + H*h + H*x**2 + g*x**3 + h*x**2 + x**4

    In [4]: Poly(expand(efac),x).all_coeffs()
    Out[4]: [1, G + g, G*g + H + h, G*h + H*g, H*h]


    Aim to factorize into two quaratics... so expand that an line up coeffs


           1 = 1 
           a = G + g 

           b = G*g + H + h     

     b - G*g = H + h 
           d = h*H


           c = G*h + H*g 
   
           G + g = a         # looks like sum of roots 
           G*g + g*g = a*g

      Subs G*g = x           # thinking of product of roots     

            
       

      Must be discrim of the quad ??  (b^2-4ac for ax**2+bx+c )

            e^2 = a^2 - b - y 
                                x**2 - a x + (b+y)/4 = 0    ## x1+x2 = a, x1*x2 = (b+y)/4
 
            f^2 = y^2/4 - d
                                x**2 + y/2 x +  d  = 0 
                               

            ef  = a*(y/4) + c/2


           h^2 - yh + d = 0 

    Vieta
            a x**2 + bx + c = 0   

    * sum of roots     : x1+x2 = -b/a  
    * product of roots :  x1x2 = c/a 

    """
    a,b,c,d,x,y,z = symbols("a:d,x:z")
    g,G,h,H = symbols("g,G,h,H")
    efac = (x**2+G*x+H)*(x**2+g*x+h)


     
    


if __name__ == '__main__':


    if 0:   
        qua = Quartic()
        #qua.iroot = 1, 2, 3, 4
        qua.iroot = 3, 2+5j, 2-5j, 1
        #qua.iroot = 1, 1, 1, 1
    pass
   
    a,b,c,d,x,y,z = symbols("a:d,x:z")
    g,G,h,H = symbols("g,G,h,H")
    efac = (x**2+G*x+H)*(x**2+g*x+h)








