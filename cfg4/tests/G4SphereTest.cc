#include <iostream>

#include "G4String.hh"
#include "G4Sphere.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);


    G4String name("sphere");

    G4double pRmin = 0. ; 
    G4double pRmax = 100. ; 
    G4double pSPhi = 0. ; 
    G4double pDPhi = 2.*CLHEP::pi ;

    G4double pSTheta = 0.f ; 
    G4double pDTheta = CLHEP::pi ;


    G4Sphere sp(name, pRmin, pRmax, pSPhi, pDPhi, pSTheta, pDTheta );


    std::cout << sp << std::endl ; 




    return 0 ; 
}
