#include "G4Sphere.hh"

#include "X4Solid.hh"
#include "X4Mesh.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    G4Sphere* sp = X4Solid::MakeSphere("demo_sphere", 100.f, 0.f); 

    std::cout << *sp << std::endl ; 

    X4Mesh* xm = new X4Mesh(sp) ; 

    LOG(info) << xm->desc() ; 

    xm->save("/tmp/X4MeshTest/X4MeshTest.gltf"); 

    return 0 ; 
}
