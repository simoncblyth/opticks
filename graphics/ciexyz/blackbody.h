#pragma once

#include <math.h>
//  extract from http://www.fourmilab.ch/documents/specrend/specrend.c


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


