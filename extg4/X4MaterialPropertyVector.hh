#pragma once

#include "X4_API_EXPORT.hh"
#include "G4MaterialPropertyVector.hh"

template <typename T> class NPY ; 

struct X4_API X4MaterialPropertyVector
{
    const G4MaterialPropertyVector*  vec ; 

    static G4MaterialPropertyVector* FromArray(const NPY<double>* arr) ;

    template <typename T> static NPY<T>* Convert(const G4MaterialPropertyVector* vec) ; 



    X4MaterialPropertyVector(const G4MaterialPropertyVector* vec_ );     

    template <typename T> NPY<T>* convert(); 
}; 





