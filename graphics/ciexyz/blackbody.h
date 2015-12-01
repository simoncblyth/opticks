#pragma once

#include <math.h>
//  extract from http://www.fourmilab.ch/documents/specrend/specrend.c

//  seems off by factor of pi ?

/*
In [9]: from scipy.constants import h,c,k

In [10]: 2.0*h*c*c
Out[10]: 1.1910428681415875e-16

In [11]: 2.0*h*c*c*np.pi
Out[11]: 3.741771524664128e-16

In [12]: h*c/k
Out[12]: 0.014387769599838155


*/


#define BB_SPECTRUM(NAME,TEMP) \
double NAME(double wavelength) \
{   \
    double bbTemp = TEMP ;  \
    double wlm = wavelength * 1e-9;   /* Wavelength in meters */  \
                   \
    return (3.74183e-16 * pow(wlm, -5.0)) /   \
           (exp(1.4388e-2 / (wlm * bbTemp)) - 1.0);  \
}  \


BB_SPECTRUM(bb5k, 5000);
BB_SPECTRUM(bb6k, 6000);


