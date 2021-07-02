/**
X4ScintillationIntegralTest
=============================

1. loads /tmp/G4OpticksAnaMgr/FASTCOMPONENT.npy 
2. performs the numerical integration just like DsG4Scintillation with X4ScintillationIntegral
3. converts to X4PhysicsOrderedFreeVector to allow persisting
4. persists to /tmp/G4OpticksAnaMgr/X4ScintillationIntegralTest.npy

::

    diff ScintillationIntegral.npy X4ScintillationIntegralTest.npy

**/
#include "X4MaterialPropertyVector.hh"
#include "X4PhysicsOrderedFreeVector.hh"
#include "X4ScintillationIntegral.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include "OPTICKS_LOG.hh"
#include "NPY.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* dir = "/tmp/G4OpticksAnaMgr" ; 
  
    const char* iname = "FASTCOMPONENT.npy" ; 

    LOG(info) << " load from  " << dir << "/" << iname ; 

    NPY<double>* fc = NPY<double>::load(dir, iname ) ; 

    if(fc == nullptr) return 0 ; 

    G4MaterialPropertyVector* theFastLightVector = X4MaterialPropertyVector::FromArray(fc) ; 

    G4PhysicsOrderedFreeVector* ScintillatorIntegral = X4ScintillationIntegral::Integral(theFastLightVector) ; 

    X4PhysicsOrderedFreeVector* xScintillatorIntegral = new X4PhysicsOrderedFreeVector(ScintillatorIntegral) ; 

    NPY<double>* si = xScintillatorIntegral->convert<double>() ; 

    const char* derived_name = "X4ScintillationIntegralTest.npy" ; 

    LOG(info) << " save to " << dir << "/" << derived_name ; 

    si->save(dir, derived_name ); 

    const char* original_name = "ScintillationIntegral.npy" ; 

    NPY<double>* si_original = NPY<double>::load(dir, original_name  ); 

    bool dump = true ; 

    unsigned mismatch_items = NPY<double>::compare( si_original, si, dump );   
   
    LOG(info) 
        << " compare original " << original_name 
        << " with derived " << derived_name 
        <<  " mismatch_items " << mismatch_items 
        ; 

    assert( mismatch_items == 0 ); 

    NPY<double>* wl = X4ScintillationIntegral::CreateWavelengthSamples( ScintillatorIntegral, 1000000 ); 
    const char* localSamples = "localSamples.npy" ; 
    LOG(info) << " save to " << localSamples ; 
    wl->save(dir, localSamples); 


    unsigned num_bins = 4096 ; 
    NPY<double>* icdf = X4ScintillationIntegral::CreateGeant4InterpolatedInverseCDF(ScintillatorIntegral, num_bins ) ; 

    const char* icdf_name = "icdf.npy" ; 
    LOG(info) << " save to " << icdf_name ; 
    icdf->save(dir, icdf_name);  

    return 0 ; 
}
