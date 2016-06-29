
// without this get allocation error in passing string to the lib
#define _HAS_ITERATOR_DEBUGGING 0

#include <cassert>
#include <string>
#include <iostream>


#include "G4Material.hh"
//#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"


//#include "PLOG.hh"
//#include "CFG4_LOG.hh"


void test_G4Material()
{
    std::cerr << " test_G4Material " << std::endl ; 
    
    G4Material* material(NULL);
    assert(material == NULL);    

    std::string sname = "TestMaterial" ;
    
    //material = new G4Material(sname, z=1., a=1.01*g/mole, density=universe_mean_density ); 
    
    std::cerr << " test_G4Material ctor " << std::endl ; 

    G4double z = 1. ;
    G4double a = 1.01*g/mole ;
    G4double density = universe_mean_density ;

    G4Material* mt = new G4Material(sname, z, a, density);

    std::cerr << " test_G4Material " << mt->GetName() << std::endl ; 

    // delete mt ;  dtor segments, same when on stack and exits scope
    // alls well if you just leak 

    std::cerr << " test_G4Material DONE " << std::endl ; 

}



void test_Al()
{
   G4double z, a, density ;  
   new G4Material("Aluminium"  , z=13., a= 26.98*g/mole, density= 2.700*g/cm3);
}



int main(int , char** argv)
{
  //  PLOG_(argc, argv);
  //  CFG4_LOG_ ;


    test_G4Material();
    test_Al();


    std::cerr << " main DONE " << argv[0] << std::endl ; 



    return 0 ;
}

