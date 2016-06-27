
#include <cassert>
#include <string>


#include "G4Material.hh"
#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"


#include "PLOG.hh"
#include "CFG4_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    CFG4_LOG_ ;


    G4Material* material(NULL);

    G4double z, a, density ; 

    std::string sname = "TestMaterial" ;

    
    material = new G4Material(sname, z=1., a=1.01*g/mole, density=universe_mean_density ); 




    return 0 ;
}

