#pragma once

#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"
class GScintillatorLib ; 
template <typename T> class NPY ; 
template <typename T> struct QTex ; 
struct QRng ; 

struct QUDARAP_API QScint
{
    static const plog::Severity LEVEL ; 

    const GScintillatorLib* lib ; 
    NPY<float>*             buf ; 
    unsigned                ni ; 
    unsigned                nj ; 
    unsigned                nk ; 
    QTex<float>*            tex ; 
    const QRng*             rng ; 

    QScint(const GScintillatorLib* lib_); 
    void init(); 

    void generate( float* wavelength, unsigned num_wavelength ); 
    void dump(     float* wavelength, unsigned num_wavelength ); 

};


