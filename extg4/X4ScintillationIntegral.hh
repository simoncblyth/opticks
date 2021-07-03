#pragma once
#include "X4_API_EXPORT.hh"

#include "plog/Severity.h"
class G4PhysicsOrderedFreeVector ; 
#include "G4MaterialPropertyVector.hh"


template <typename T> class NPY ; 

struct X4_API X4ScintillationIntegral
{
    static const plog::Severity LEVEL ; 

    static G4PhysicsOrderedFreeVector* Integral( const G4MaterialPropertyVector* theFastLightVector ) ;
    static NPY<double>* CreateWavelengthSamples(            const G4PhysicsOrderedFreeVector* ScintillatorIntegral, G4int num_samples=1000000 ) ;
    static NPY<double>* CreateGeant4InterpolatedInverseCDF( const G4PhysicsOrderedFreeVector* ScintillatorIntegral, unsigned num_bins, unsigned hd_factor, const char* name ); 

}; 


