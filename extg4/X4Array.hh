#pragma once

#include <string>
#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"
#include "G4Types.hh"


struct NP ; 
template <typename T> class NPY ; 

class G4PhysicsVector ; 

/**
X4Array : formerly X4PhysicsOrderedFreeVector
==================================================

This class was formerly called *X4PhysicsOrderedFreeVector* but as
the functionality added (persisting and conversion from NPY and NP arrays)
does not depend on the values or domain being monotonic this is now named *X4Array*
despite being "paired" with G4PhysicsVector.
The name "X4PhysicsVector" is taken already for a somewhat different purpose. 

Geant4 Background
-------------------

G4PhysicsOrderedFreeVector inherits from G4PhysicsVector, with no 
additional member data only additional methods profiting from monotonic 
energy domain.

The crucial thing that G4PhysicsOrderedFreeVector adds on top of G4PhysicsVector
is GetEnergy which returns the domain (x) corresponding to a value (y), 
however that requires the values to be monotonic. 

**/

struct X4_API X4Array 
{
    static const plog::Severity LEVEL ; 
    static std::string Desc(const G4PhysicsVector* vec );

    template<typename T> 
    static NPY<T>* Convert(const G4PhysicsVector* vec) ;

    static X4Array* Load(const char* base, const char* name=nullptr, double en_scale=0. ); 
    static X4Array* FromArray(const NP* a ); 
    static X4Array* FromArray(const NPY<double>* a );    
    static X4Array* FromPairData(const double* pdata, unsigned npair ); 


    const NPY<double>*          src ;
    const NP*                   asrc ;   
    G4PhysicsVector*            vec ; 


    X4Array( G4PhysicsVector* vec_ ) ; 
    std::string desc() const ;

    G4double getMidBinValue() const ;
    void changeAllToMidBinValue();
    void putValues( G4double value ); 

    template<typename T> NPY<T>* convert() ;

};



