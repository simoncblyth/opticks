#pragma once

#include <string>
#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"
#include "G4types.hh"

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

    G4PhysicsOrderedFreeVector* vec ; 

    X4PhysicsOrderedFreeVector( G4PhysicsOrderedFreeVector* vec_ ) ; 
    std::string desc() const ;

    G4double getMidBinValue() const ;
    void changeAllToMidBinValue();
    void putValues( G4double value ); 

};



