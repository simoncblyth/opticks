#pragma once
/**
X4Scintillation
================


Before 1100::

    typedef G4PhysicsOrderedFreeVector G4MaterialPropertyVector;

From 1100::

    typedef G4PhysicsFreeVector G4MaterialPropertyVector;

And from 1100 the "Ordered" methods have been consolidated into G4PhysicsFreeVector
and the class G4PhysicsOrderedFreeVector is dropped.
Try to cope with this without version barnching using edit::

   :%s/PhysicsOrderedFree/MaterialProperty/gc

Maybe will need to add some casts too.

**/
#include "X4_API_EXPORT.hh"

#include "plog/Severity.h"
#include "G4MaterialPropertyVector.hh"  

template <typename T> class NPY ; 

struct X4_API X4Scintillation
{
    static const plog::Severity LEVEL ; 

    const NPY<double>* fast ;
    const NPY<double>* slow ;
    const double epsilon ; 
    unsigned mismatch ; 

    const G4MaterialPropertyVector* theFastLightVector ; 
    const G4MaterialPropertyVector* theSlowLightVector ; 
    const G4MaterialPropertyVector* ScintillationIntegral ; 

    X4Scintillation( const NPY<double>* fast_, const NPY<double>* slow_ ); 

    NPY<double>* createWavelengthSamples( unsigned num_samples=1000000 ); 
    NPY<double>* createGeant4InterpolatedInverseCDF( unsigned num_bins=4096, unsigned hd_factor=20, const char* material_name="LS", bool energy_not_wavelength=false ); 


    static G4MaterialPropertyVector* Integral( const G4MaterialPropertyVector* theFastLightVector ) ;
    static NPY<double>* CreateWavelengthSamples(            const G4MaterialPropertyVector* ScintillatorIntegral, unsigned num_samples ) ;
    static NPY<double>* CreateGeant4InterpolatedInverseCDF( const G4MaterialPropertyVector* ScintillatorIntegral, unsigned num_bins, unsigned hd_factor, const char* name, bool energy_not_wavelength ); 

}; 


