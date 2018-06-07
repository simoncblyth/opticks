
#include "G4Sphere.hh"
#include "X4Solid.hh"
#include "OPTICKS_LOG.hh"

G4Sphere* make_sphere()
{
    G4String name("sphere");
    G4double pRmin = 0. ; 
    G4double pRmax = 100. ; 
    G4double pSPhi = 0. ; 
    G4double pDPhi = 2.*CLHEP::pi ;

    G4double pSTheta = 0.f ; 
    G4double pDTheta = CLHEP::pi ;

    G4Sphere* sp = new G4Sphere(name, pRmin, pRmax, pSPhi, pDPhi, pSTheta, pDTheta );
    std::cout << *sp << std::endl ; 
    return sp ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    
    G4Sphere* sp = make_sphere(); 

    X4Solid* so = X4Solid::Create(sp) ; 

    LOG(info) << so->desc() ; 

 
    return 0 ; 
}
