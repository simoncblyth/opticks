#include <iomanip>
#include <sstream>

#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4ThreeVector.hh"

#include "CThreeVector.hh"
#include "CEvent.hh"

/**

Primaries are converted from HepMC::GenEvent into G4Event
in LSExpPrimaryGeneratorAction::GeneratePrimaries for example.

**/


unsigned CEvent::GetNumberOfPrimaryOpticalPhotons(const G4Event* event)   // static 
{
    return 0u ; 
}


std::string CEvent::DescPrimary(const G4Event* event)   // static 
{
    std::stringstream ss ; 
    G4int numPrim = event->GetNumberOfPrimaryVertex(); 
    ss 
        << "CEvent::DescPrimary"
        << " numPrim " << numPrim 
        ;

    for(int i=0 ; i < numPrim ; i++)
    {
        G4PrimaryVertex* vtx = event->GetPrimaryVertex(i) ; 
        G4ThreeVector pos = vtx->GetPosition();
        ss 
            << " primaryVertex " << std::setw(2)  << i 
            << " pos " << CThreeVector::Format(pos)         
            ;

    }
    std::string s = ss.str(); 
    return s ; 
}






