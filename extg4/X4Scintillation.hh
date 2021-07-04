#pragma once
#include "X4_API_EXPORT.hh"

#include "plog/Severity.h"

#include "G4MaterialPropertyVector.hh"   // just a typedef 
class G4PhysicsOrderedFreeVector ; 

template <typename T> class NPY ; 

struct X4_API X4Scintillation
{
    static const plog::Severity LEVEL ; 

    const NPY<double>* fast ;
    const NPY<double>* slow ;
    unsigned mismatch ; 

    const G4MaterialPropertyVector* theFastLightVector ; 
    const G4MaterialPropertyVector* theSlowLightVector ; 
    const G4PhysicsOrderedFreeVector* ScintillationIntegral ; 

    X4Scintillation( const NPY<double>* fast_, const NPY<double>* slow_ ); 

    NPY<double>* createWavelengthSamples( unsigned num_samples=1000000 ); 
    NPY<double>* createGeant4InterpolatedInverseCDF( unsigned num_bins=4096, unsigned hd_factor=20, const char* material_name="LS" ); 


    static G4PhysicsOrderedFreeVector* Integral( const G4MaterialPropertyVector* theFastLightVector ) ;
    static NPY<double>* CreateWavelengthSamples(            const G4PhysicsOrderedFreeVector* ScintillatorIntegral, unsigned num_samples ) ;
    static NPY<double>* CreateGeant4InterpolatedInverseCDF( const G4PhysicsOrderedFreeVector* ScintillatorIntegral, unsigned num_bins, unsigned hd_factor, const char* name ); 

}; 


