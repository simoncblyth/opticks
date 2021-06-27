#include "X4MaterialPropertyVector.hh"
#include "X4PhysicsOrderedFreeVector.hh"
#include "X4ScintillationIntegral.hh"

#include "OPTICKS_LOG.hh"
#include "NPY.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* dir = "/tmp/G4OpticksAnaMgr" ; 
  
    const char* iname = "FASTCOMPONENT.npy" ; 

    LOG(info) << " load from  " << dir << "/" << iname ; 

    NPY<double>* fc = NPY<double>::load(dir, iname ) ; 

    G4MaterialPropertyVector* theFastLightVector = X4MaterialPropertyVector::FromArray(fc) ; 

    G4PhysicsOrderedFreeVector* ScintillatorIntegral = X4ScintillationIntegral::Integral(theFastLightVector) ; 

    X4PhysicsOrderedFreeVector* xScintillatorIntegral = new X4PhysicsOrderedFreeVector(ScintillatorIntegral) ; 

    NPY<double>* si = xScintillatorIntegral->convert<double>() ; 

    const char* oname = "X4ScintillationIntegralTest.npy" ; 

    LOG(info) << " save to " << dir << "/" << oname ; 

    si->save(dir, oname ); 

    return 0 ; 
}
