#include <iomanip>
#include <sstream>

#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4ThreeVector.hh"

#include "CPrimaryVertex.hh"
#include "CThreeVector.hh"
#include "CEvent.hh"

/**

Primaries are converted from HepMC::GenEvent into G4Event
in LSExpPrimaryGeneratorAction::GeneratePrimaries for example.

**/

unsigned CEvent::NumberOfInputPhotons(const G4Event* event)   // static 
{
    G4int numPrim = event->GetNumberOfPrimaryVertex(); 
    unsigned numberOfInputPhotons = 0u ; 
    for(int i=0 ; i < numPrim ; i++)
    {
        G4PrimaryVertex* vtx = event->GetPrimaryVertex(i) ; 
        if(CPrimaryVertex::IsInputPhoton(vtx))  numberOfInputPhotons += 1 ; 
    }
    return numberOfInputPhotons ; 
}


std::string CEvent::DescPrimary(const G4Event* event)   // static 
{
    std::stringstream ss ; 
    G4int numPrim = event->GetNumberOfPrimaryVertex(); 
    unsigned numberOfInputPhotons = NumberOfInputPhotons(event); 
    ss 
        << "CEvent::DescPrimary"
        << " numPrim " << numPrim 
        << " numberOfInputPhotons " << numberOfInputPhotons
        << std::endl 
        ;

    for(int i=0 ; i < numPrim ; i++)
    {
        G4PrimaryVertex* vtx = event->GetPrimaryVertex(i) ; 
        ss 
            << std::setw(2) << i << " "  
            << CPrimaryVertex::Desc(vtx) 
            << std::endl
            ; 
    }
    std::string s = ss.str(); 
    return s ; 
}






