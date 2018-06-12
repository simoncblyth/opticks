#include <cassert>
#include "G4Sphere.hh"

#include "X4Solid.hh"
#include "X4SolidList.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);
    
    G4Sphere* s1 = X4Solid::MakeSphere("s1", 100.f, 0.f); 
    G4Sphere* s2 = X4Solid::MakeSphere("s2", 100.f, 0.f); 

    LOG(info) << "s1:\n" << *s1 ; 
    LOG(info) << "s2:\n" << *s2 ; 


    X4SolidList sl ; 

    sl.addSolid(s1); 
    sl.addSolid(s1); 
    sl.addSolid(s1); 

    sl.addSolid(s2); 

    assert( sl.getNumSolids() == 2 ); 

    LOG(info) << sl.desc() ; 
 
    return 0 ; 
}
