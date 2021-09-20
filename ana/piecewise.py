#!/usr/bin/env python
"""
piecewise.py
==============

::

    ipython -i piecewise.py


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



QCerenkov : Whacky Parabolic Ideas
-------------------------------------

* https://en.wikipedia.org/wiki/Simpson%27s_rule
* https://en.wikipedia.org/wiki/Adaptive_Simpson%27s_method

Looks like storing the mid-bin value of the cumulative integral 
would be sufficient to give you the parabola. Allowing in principal
to recover the equation of the parabola from 3 points and giving 
energy lookup. 

* y = a x^2 + b x + c 

* 3 points -> 3 equations in 3 unknowns (a,b,c) : matrix inversion gives you (a,b,c)


                  2   
             .    
          .  
        1
      .  
    0


Obtaining piecewise parabolic cumulative integral by storing 
(a,b,c) parameters of the parabola ? 
Which would avoid the need for loadsa bins.

* TODO: explore this by ana/piecewise.py sympy expts handling 
  each bin separately to avoid symbolic integration troubles
  and construct the symbolic cumulative integral   

* hmm: better to solve once and store (a,b,c) for each bin 
* then can do energy lookup by solving quadratic, it will be monotonic
  so no problem of picking the parabolic piece applicable and then 
  picking the root ?









"""
import logging, os
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt 

from opticks.ana.edges import divide_bins

from sympy.plotting import plot
from sympy import Piecewise, piecewise_fold, Symbol, Interval, integrate, Max, Min
from sympy import lambdify, ccode 
from sympy.utilities.codegen import codegen

from sympy.abc import x, y
from sympy.solvers import solve

e = Symbol('e', positive=True)



from opticks.ana.bisect import bisect


def ri_load():
    ri_approx = np.array([
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

    ri_path = os.path.expandvars("$OPTICKS_KEYDIR/GScintillatorLib/LS_ori/RINDEX.npy")
    if os.path.exists(ri_path):
        log.info("load from %s " % ri_path)
        ri = np.load(ri_path) 
        ri[:,0] *= 1e6      # MeV to eV 
    else:
        log.fatal("default to approximate ri")
        ri = ri_approx
    pass
    return ri 
pass

ri = ri_load()


def make_piece(i, b=1.5):
    """
    Max(,0.) complicates the integrals obtained, 
    but is a great simplification in that do not need to manually control 
    the range to skip regions where s2 dips negative 

    Note that with b=1.5 the the last piece from make_piece(16)
    reduces to zero because v0 = v1 which are both -ve 
    and sympy succeeds to simplify to zero.  

    Curiously make_piece(0) does not similarly reduce to zero
    although it could do. 
    """
    assert i >= 0 
    assert i <= len(ri)-1


    e0, r0 = ri[i]
    e1, r1 = ri[i+1]

    v0 = ( 1 - b/r0 ) * ( 1 + b/r0 )
    v1 = ( 1 - b/r1 ) * ( 1 + b/r1 )

    fr = (e-e0)/(e1-e0)   # e = e0, fr=0,  e = e1, fr=1 
    pt = ( Max(v0*(1-fr) + v1*fr,0),  (e > e0) & (e < e1) )    
    ot = (0, True )

    pw = Piecewise( pt, ot ) 
    return pw



def get_check(BetaInverse=1.55):
    ck = np.zeros_like(ri)  
    ck[:,0] = ri[:,0]
    ck[:,1] = (1. - BetaInverse/ri[:,1])*(1.+BetaInverse/ri[:,1])    
    return ck 


class S2Integral(object):
    """
    Attemps to treat BetaInverse symbolically rather than as a constant cause integration to fail
    or just not happen ... kinda makes sense the BetaInverse has a drastic effect on the s2 
    so seems that must subs it before integrating. Or for maximum simplicity just handle as float constant. 

    BUT: the integrals are just parabolas and the BetaInverse just changes the coefficients
    of the piecewise linerar s2 so with patience could write down the piecewise integral function 
    with the BetaInverse still symbolic.  The problem is the Max(_,0.)  

    * https://math.stackexchange.com/questions/3714720/if-f-and-g-are-both-continuous-then-the-functions-max-f-g-and-min

    ::
                         a + b + | a - b |                         a + |a|
         max( a, b ) =  --------------------      max( a, 0 ) =   ----------
                                 2                                    2 

                          a + b - | a - b |                        a - |a|
         min( a, b ) =   --------------------     min( a, 0 ) =   ------------
                                 2                                    2 


         max( a, b ) + min( a, b) = a + b        max(a, 0) + min(a, 0) = a 
    


    """
    FINE_STRUCTURE_OVER_HBARC_EVMM = 36.981

    def __init__(self, BetaInverse=1.55, symbolic_add=False):
        self.b = BetaInverse

        s2 = {}
        is2 = {}

        emn = ri[0,0]
        emx = ri[-1,0]
        emd = (emn+emx)/2.

        log.info("s2, is2...")
        for i in range( len(ri)-1 ):
            s2[i] = make_piece( i, b=BetaInverse ) 
            is2[i] = integrate( s2[i], e )   # do integral for each piece separately as doesnt work when added together
            print("s2[%d]" % i, s2[i])
            print("is2[%d]" % i, is2[i])
            pass
        pass

        if symbolic_add:
            log.info("[ is2a...")   # hmm symbolically adding up the pieces takes a long time 
            is2a = sum(is2.values())  
            log.info("] is2a...")

            norm = is2a.subs(e, emx)
            is2n = is2a/norm
            pass
            log.info("lambdify")  
            g = lambdify(e, is2n, "numpy")    
            vg = np.vectorize( g ) 
            # lambdify generated lambda does not support numpy array aguments in sympy 1.6.2, docs suggest it does on 1.8, hence vectorize   
        else:
            is2a = None
            is2n = None
            norm = None
            g = None
            vg = None
        pass

        qwns = "s2 is2 is2a norm is2n emn emx emd g vg".split()
        for qwn in qwns: 
            setattr(self, qwn, locals()[qwn])  
        pass

    def is2c(self, ee):
         return self.vg(ee)

    def cliff_plot(self):
        is2 = self.is2
        emn = self.emn 
        emx = self.emx 

        plot( *is2.values(), (e, emn, emx), show=True, adaptive=False, nb_of_points=500)   # cliff edges from each bin  

    def s2_plot(self):
        is2n = self.is2n
        emn = self.emn 
        emx = self.emx 

        plot(is2n[BetaInverse], (e, emn, emx), show=True )

    def energy_lookup(self, u=0.5):
        is2n = self.is2n
        emn = self.emn 
        emx = self.emx 

        fn = lambda x:is2n.subs(e, x) - u 
        en = bisect( fn, emn, emx, 20 )         # energy lookup 
        return en 

    def is2a_(self, en):
        """
        multiplying by a float constant slows sympy down so do scaling here 
        """
        is2a = self.is2a
        Rfact = self.FINE_STRUCTURE_OVER_HBARC_EVMM 
        return Rfact*is2a.subs(e, en)

    def is2sum_(self, en):
        """
        try numerical adding instead of symbolic adding  
        """
        is2 = self.is2
        Rfact = self.FINE_STRUCTURE_OVER_HBARC_EVMM 
        tot = 0. 
        for i in range( len(ri)-1 ):
             tot += Rfact*is2[i].subs(e, en)
        pass
        return tot


    def export_as_c_code(self, name="s2integral", base="/tmp"):
        """ 
        # see ~/np/tests/s2integralTest.cc for testing the generated code 
        """
        is2n = self.is2n
        expr = is2n 
        cg = codegen( (name, expr), "C99", name, header=True )
        open(os.path.join(base, cg[0][0]), "w").write(cg[0][1]) 
        open(os.path.join(base, cg[1][0]), "w").write(cg[1][1]) 


    def plot_is2c(self):
        emn = self.emn 
        emx = self.emx 
        ee = np.linspace( emn, emx, 200 )
        ie = self.is2c(ee)
        BetaInverse = self.b 

        title = "ana/piecewise.py : plot_is2c : BetaInverse %7.4f " % BetaInverse

        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        fig.suptitle(title)
        ax.plot( ee, ie, label="plot_is2c" )
        ax.legend()
        fig.show() 

    @classmethod
    def Scan(cls, tt, mul=2):
        """
        see QUDARap/tests/QCerenkovTest.cc .py for comparison with numerical integral  
        """
        ee = divide_bins(ri[:,0], mul)   
        pass
        sc = np.zeros( (len(tt), len(ee), 2) )
        for i, k in enumerate(sorted(tt)):
            t = tt[k]
            BetaInverse = t.b 
            for j,en in enumerate(ee):
                is2sum_val = t.is2sum_(en)
                sc[i, j] = en, is2sum_val 
                print("BetaInverse %10.5f en %10.5f is2sum_val : %10.5f " % (BetaInverse, en, is2sum_val) )
            pass
        pass
        path = "/tmp/ana/piecewise/scan.npy"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        pass
        log.info("save scan to %s " % path )
        np.save(path, sc)
        return sc


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)    

    plt.ion()

    #bis_ = "1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.792"
    bis_ = "1"
    bis = list(map(float, bis_.split()))

    mul = 2 

    tt = {}
    for BetaInverse in bis:
        tt[BetaInverse] = S2Integral(BetaInverse=BetaInverse)
    pass
 
    #t = tt[1.]
    #t.cliff_plot()
    #t.plot_is2c()    
    #t.export_as_c_code()
    #is2a_val = t.is2a_(t.emx)

    S2Integral.Scan(tt)


