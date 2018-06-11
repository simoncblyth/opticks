#include "G4Sphere.hh"
#include "X4Mesh.hh"
#include "X4Solid.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    G4Sphere* sp = X4Solid::MakeSphere("demo_sphere"); 
    std::cout << *sp << std::endl ; 

    X4Mesh* mh = new X4Mesh(sp) ; 

    LOG(info) << mh->desc() ; 

 
    return 0 ; 
}
