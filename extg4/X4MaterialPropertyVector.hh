#pragma once

#include "X4_API_EXPORT.hh"
#include "G4MaterialPropertyVector.hh"

struct NP ; 
template <typename T> class NPY ; 

/**
X4MaterialPropertyVector
=========================

Simple direct convert and wrappers to isolate Geant4 API changes

1. no interpolation
2. no mapping of energy to wavelength.

**/

struct X4_API X4MaterialPropertyVector
{
    const G4MaterialPropertyVector*  vec ; 

    // wrappers to isolate Geant4 API changes with version, particularly the transition to 1100
    static G4double GetMinLowEdgeEnergy( const G4MaterialPropertyVector* vec ); 
    static G4double GetMaxLowEdgeEnergy( const G4MaterialPropertyVector* vec ); 
    static G4bool   IsFilledVectorExist( const G4MaterialPropertyVector* vec ); 

    // convenience conversions
    static G4MaterialPropertyVector* FromArray(const NP* a) ;
    static G4MaterialPropertyVector* FromArray(const NPY<double>* arr) ;

    X4MaterialPropertyVector(const G4MaterialPropertyVector* vec_ );     

    G4double GetMinLowEdgeEnergy() const ; 
    G4double GetMaxLowEdgeEnergy() const ; 
    G4bool   IsFilledVectorExist() const ; 

   
    template <typename T> NPY<T>* convert(); 
    template <typename T> static NPY<T>* Convert(const G4MaterialPropertyVector* vec) ; 

    static NP* ConvertToArray(const G4MaterialPropertyVector* vec); 

}; 





