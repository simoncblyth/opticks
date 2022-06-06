
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"

#include "G4StepPoint.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "U4StepPoint.hh"

/**
U4StepPoint::Update
---------------------

* cf CWriter::writeStepPoint_

**/

void U4StepPoint::Update(sphoton& photon, const G4StepPoint* point)  // static
{
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& mom = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    
    photon.pos.x = pos.x(); 
    photon.pos.y = pos.y(); 
    photon.pos.z = pos.z(); 
    photon.time  = time/ns ; 

    photon.mom.x = mom.x(); 
    photon.mom.y = mom.y(); 
    photon.mom.z = mom.z(); 
    //photon.iindex = 0u ; 

    photon.pol.x = pol.x(); 
    photon.pol.y = pol.y(); 
    photon.pol.z = pol.z(); 
    photon.wavelength = wavelength/nm ; 
}

