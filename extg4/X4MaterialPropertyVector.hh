#pragma once

#include "X4_API_EXPORT.hh"
#include "G4MaterialPropertyVector.hh"

struct NP ; 
template <typename T> class NPY ; 

/**
X4MaterialPropertyVector
=========================

Simple direct convert:

1. no interpolation
2. no mapping of energy to wavelength.

**/

struct X4_API X4MaterialPropertyVector
{
    const G4MaterialPropertyVector*  vec ; 

    static G4MaterialPropertyVector* FromArray(const NP* a) ;
    static G4MaterialPropertyVector* FromArray(const NPY<double>* arr) ;

    template <typename T> static NPY<T>* Convert(const G4MaterialPropertyVector* vec) ; 
    static                       NP* ConvertToArray(const G4MaterialPropertyVector* vec); 

    X4MaterialPropertyVector(const G4MaterialPropertyVector* vec_ );     

    template <typename T> NPY<T>* convert(); 
}; 





