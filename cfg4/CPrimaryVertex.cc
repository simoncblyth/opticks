#include <iomanip>
#include <sstream>

#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "CThreeVector.hh"
#include "CPrimaryVertex.hh"


std::string CPrimaryVertex::Desc(const G4PrimaryVertex* vtx )
{
    G4int numberOfParticle = vtx->GetNumberOfParticle() ; 
    G4ThreeVector pos = vtx->GetPosition();
    G4double t0 = vtx->GetT0();

    std::stringstream ss ; 
    ss 
        << "CPrimaryVertex::Desc"
        << " t0 " << std::fixed << std::setw(10) << std::setprecision(3) << t0 
        << " pos " << CThreeVector::Format(pos)         
        << " numberOfParticle " << std::setw(2) << numberOfParticle
        ;

    if( numberOfParticle > 1 ) ss << std::endl ; 

    for(int i=0 ; i < numberOfParticle ; i++ )
    {
        const G4PrimaryParticle* prim = vtx->GetPrimary(i) ; 
        G4int pdgCode = prim->GetPDGcode()  ; 

        const G4ThreeVector& dir = prim->GetMomentumDirection(); 
        G4double kineticEnergy = prim->GetKineticEnergy() ; 
        G4double wavelength = h_Planck*c_light/kineticEnergy ; 

        ss 
            << " pdgCode " << std::setw(7) << pdgCode 
            << " dir " << CThreeVector::Format(dir) 
            << " kEn " << kineticEnergy 
            << " nm "  << wavelength/nm 
            ;

       if( numberOfParticle > 1 ) ss << std::endl ; 
    }

    std::string s = ss.str(); 
    return s;  
}

