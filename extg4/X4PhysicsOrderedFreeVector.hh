#pragma once

#include <string>
#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"
#include "G4Types.hh"


struct NP ; 
template <typename T> class NPY ; 

class G4PhysicsOrderedFreeVector ; 

/**
X4PhysicsOrderedFreeVector
===============================

G4PhysicsOrderedFreeVector inherits from G4PhysicsVector

**/

struct X4_API X4PhysicsOrderedFreeVector 
{
    static const plog::Severity LEVEL ; 
    static std::string Desc(const G4PhysicsOrderedFreeVector* vec );

    template<typename T> 
    static NPY<T>* Convert(const G4PhysicsOrderedFreeVector* vec) ;

    static X4PhysicsOrderedFreeVector* Load(const char* base, const char* name=nullptr, double en_scale=0. ); 
    static X4PhysicsOrderedFreeVector* FromArray(const NP* a ); 
    static X4PhysicsOrderedFreeVector* FromArray(const NPY<double>* a );    
    static X4PhysicsOrderedFreeVector* FromPairData(const double* pdata, unsigned npair ); 


    const NPY<double>*          src ;
    const NP*                   asrc ;   
    G4PhysicsOrderedFreeVector* vec ; 


    X4PhysicsOrderedFreeVector( G4PhysicsOrderedFreeVector* vec_ ) ; 
    std::string desc() const ;

    G4double getMidBinValue() const ;
    void changeAllToMidBinValue();
    void putValues( G4double value ); 

    template<typename T> NPY<T>* convert() ;

};



