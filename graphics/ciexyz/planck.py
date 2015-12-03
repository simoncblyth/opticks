#!/usr/bin/env python
"""

Generated photons feature a plateau from 80nm to 205nm followed 
by step up to join the curve.  

* pushing out generation to 10M, 100M doesnt change this feature
  other than scaling the plateau level

* low bins of pk.pdf feature very small numbers..., 
  attempt to avoid numerical issues using arbitrary scaling moves the cut 
  to 210nm, also a step structure in the generated distribution is 
  apparent 

* problem presumably from too great a probability range... so that 
  never get to generate any at lower bins somehow convolved with 
  numerical precision to cause steps ? 

::

    In [11]: pk.cdf[0]
    Out[11]: 0.0

    In [12]: pk.cdf[1]
    Out[12]: 1.0708856422955714e-17

    In [13]: pk.cdf[-1]
    Out[13]: 0.99999999999999045


    In [18]: u.min()
    Out[18]: 3.6289631744068629e-07

    In [19]: u.max()
    Out[19]: 0.99999976714026029

    In [20]: u[u>3.6289631744068629e-07].min()
    Out[20]: 1.9915794777780604e-06


Huh 200nm is on the plateau but the cdf is not outrageously small there ?::

    In [27]: pk.cdf[15001]
    Out[27]: 0.0065557782251502248

    In [28]: np.where(np.logical_and(w > 200.,w<200.1) )
    Out[28]: (array([15001, 15002, 15003, 15004, 15005, 15006, 15007, 15008, 15009, 15010]),)


Moving the low wavelength up from 80nm to 200nm avoids the objectionable plateau and cliff,
but there is still a stepping structure though.


**RESOLVED** 

    The interpolation was using too coarse a binning that 
    worked fine for most of the distribution, but not good enough 
    for the low wavelength turn on 




* http://www.yale.edu/ceo/Documentation/ComputingThePlanckFunction.pdf

::


             2hc^2 w^-5
   Bw(T) =   ----------------
              e^(beta) - 1


             hc/kT
    beta =   -----
              w


Where e^(beta) >> 1  (Wien approximation) Plank tends to 


   Bw(T) = 2hc^2 w^-5 e^(-beta)
             




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


def planck(nm, K=6500., arbitrary=True):
    if arbitrary:
       wav = nm
       a = np.power(200,5)
    else:
       wav = nm/1e9 
       a = 2.0*h*c*c
    pass

    hc_over_kT = (h*c)/(k*K)

    b = hc_over_kT*1e9/nm

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
    def __init__(self, w, K=6500, N_idom=4096):

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
        
        idom = np.linspace(0,1, N_idom)  
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
        self.u = u
        self.gen = gen 
        return gen



def cf_gsrclib():
    sl = np.load("/tmp/gsrclib.npy")
    return sl[0,:,0]
    


if __name__ == '__main__':

    plt.ion()

    w = np.arange(80.,801,.1, dtype=np.float64)

    pk = Planck(w, K=6500)

    u = np.random.rand(1e6)

    gen = pk(u)    
 

    nm = 100
    wb = w[::nm]
    hn, hd = np.histogram(gen, wb)
    assert np.all(hd == wb)
    assert len(hn) == len(wb) - 1   # looses one bin 

    s_avg = pk.pdf * hn.sum() * nm


    # hmm lands spot on with -post

    plt.plot( hd[:-1], hn , drawstyle="steps-post")  # -pre -mid -post

    plt.plot( pk.mid, s_avg ) 

    plt.axis([w.min()-100, w.max()+100, 0, s_avg.max()*1.1])


    plt.show()




