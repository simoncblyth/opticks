#!/usr/bin/env python
"""

* http://www.yale.edu/ceo/Documentation/ComputingThePlanckFunction.pdf

::


             2hc^2 w^-5
   Bw(T) =   ----------------
              e^(beta) - 1


             hc/kT
    beta =   -----
              w



Need inverse CDF of Planck to put into texture
follow ggeo-/GProperty<T>::createInverseCDF

* even domain on 0:1 (needed for quick texture lookup)
* sample the CDF across this domain  

* env/geant4/geometry/collada/g4daeview/sample_cdf.py

"""

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.constants import h,c,k
except ImportError:
    h = 6.62606957e-34
    c = 299792458.0
    k = 1.3806488e-23


def planck(nm, K=6500.):
    wav = nm/1e9 
    a = 2.0*h*c*c
    b = (h*c)/(k*K)/wav
    return a/(np.power(wav,5) * np.expm1(b))  


def planck_plot():
    w = np.arange(100,1000,1)
    for temp in np.arange(4000,10000,500):
        intensity = planck(w, temp)
        plt.plot(w, intensity, '-') 



def construct_inverse_cdf_0( bva, bdo,  N ):
    """
    :param bva: basis values normalized to unity
    :param bdo: basis domain 
    :param N: number of bins to use across 0:1 range

    Note the np.interp xp<->fp inversion,
    
    (x)    ido: freshly minted 0:1 domain  
    (xp)  cbva: monotonic CDF in range 0:1, 
    (fp)   bdo: basis domain  

    """
    assert np.allclose( bva.sum(), 1.0 )
    assert len(bva) == len(bdo)

    cbva = np.cumsum(bva)   # NB no careful mid bin handling yet

    ido = np.linspace(0,1,N)  
    
    iva = np.interp( ido, cbva, bdo )  
  
    return iva, ido 



class Planck(object):
    def __init__(self, w, K=6500, N=100):

        bb = planck(w, K=K)

        # interpret into bins (so one less entry)
        avg = (bb[:-1]+bb[1:])/2.   # average of bin end values
        wid = np.diff(w)            # bin width
        mid = (w[:-1]+w[1:])/2.     # bin middle

        pdf = avg*wid

        pdf /= pdf.sum()  # NB norming later avoids last bin excursion in generated distribution

        dom = w
        cdf = np.empty(len(dom))
        cdf[0] = 0.                        # lay down zero probability 1st bin 
        np.cumsum(pdf, out=cdf[1:])
        
        idom = np.linspace(0,1,N)  
        icdf = np.interp( idom, cdf, dom )  

 
        self.avg = avg
        self.wid = wid
        self.mid = mid
        self.pdf = pdf

        self.cdf = cdf
        self.dom = dom

        self.idom = idom
        self.icdf = icdf


    def __call__(self, u ):
        gen = np.interp( u, self.idom, self.icdf )   
        return gen



def cf_gsrclib():
    sl = np.load("/tmp/gsrclib.npy")
    return sl[0,:,0]
    


if __name__ == '__main__':

    plt.ion()

    w = np.arange(300,801,.1, dtype=np.float64)

    pk = Planck(w, K=6500)

    u = np.random.rand(1e6)

    gen = pk(u)    
 

    nm = 50
    wb = w[::nm]
    hn, hd = np.histogram(gen, wb)
    assert np.all(hd == wb)
    assert len(hn) == len(wb) - 1   # looses one bin 

    s_avg = pk.pdf * hn.sum() * nm



    plt.plot( hd[:-1], hn , drawstyle="steps")

    plt.plot( pk.mid, s_avg ) 

    plt.axis([w.min()-100, w.max()+100, 0, s_avg.max()*1.1])


    plt.show()




