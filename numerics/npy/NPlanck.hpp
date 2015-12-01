#pragma once

#include <math.h>

/*
//  extract from http://www.fourmilab.ch/documents/specrend/specrend.c
//  http://physics.info/planck/


n [9]: from scipy.constants import h,c,k

In [10]: 2.0*h*c*c
Out[10]: 1.1910428681415875e-16

In [11]: 2.0*h*c*c*np.pi
Out[11]: 3.741771524664128e-16

In [12]: h*c/k
Out[12]: 0.014387769599838155


 41 def planck(nm, K):
 42     wav = nm/1e9
 43     a = 2.0*h*c*c
 44     b = (h*c)/(k*K)/wav
 45     return a/(np.power(wav,5) * np.expm1(b))
 46 

*/

double planck_spectral_radiance(double nm, double K) 
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



