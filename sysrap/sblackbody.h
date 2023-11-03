#pragma once

/**
sblackbody.h
===============



* from the former npy/NPlanck.cpp

::

    In [1]: from scipy.constants import h,c,k
    In [2]: h,c,k 
    Out[2]: (6.62607015e-34, 299792458.0, 1.380649e-23)

* https://physics.info/planck/
* http://www.fourmilab.ch/documents/specrend/specrend.c
* https://en.wikipedia.org/wiki/Planck%27s_law

TODO RESCUE::

   npy/ciexyz.h 


**/

#include <cmath>

struct sblackbody
{
    static double planck_spectral_radiance(double nm, double K=6500.) ; 
};

inline double sblackbody::planck_spectral_radiance(double nm, double K)  
{   
    double h = 6.62606957e-34 ;
    double c = 299792458.0 ;
    double k = 1.3806488e-23 ;

    double a = 2.0*h*c*c ; 
    double b = h*c/k ;
    
    double wlm = nm * 1e-9;   
  
    return (a * pow(wlm, -5.0)) /   
           (exp(b / (wlm * K)) - 1.0);  
}  

