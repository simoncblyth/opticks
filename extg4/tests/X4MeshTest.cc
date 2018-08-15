#include "G4Sphere.hh"

#include "X4Solid.hh"
#include "X4Mesh.hh"

#include "OPTICKS_LOG.hh"


void test_convert_save()
{
    G4Sphere* sp = X4Solid::MakeSphere("demo_sphere", 100.f, 0.f); 

    std::cout << *sp << std::endl ; 

    X4Mesh* xm = new X4Mesh(sp) ; 

    LOG(info) << xm->desc() ; 

    xm->save("/tmp/X4MeshTest/X4MeshTest.gltf"); 
}


void test_placeholder()
{
    G4Sphere* sp = X4Solid::MakeSphere("demo_sphere", 100.f, 0.f); 

    GMesh* pl = X4Mesh::Placeholder(sp );
 
    assert( pl ); 

}

/*

(lldb) bt
* thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=EXC_I386_GPFLT)
  * frame #0: 0x000000010248dd91 libG4geometry.dylib`G4Sphere::DistanceToOut(CLHEP::Hep3Vector const&, CLHEP::Hep3Vector const&, bool, bool*, CLHEP::Hep3Vector*) const + 161
    frame #1: 0x000000010010c95e libExtG4.dylib`X4Mesh::Placeholder(solid=0x00000001068147d0) at X4Mesh.cc:38
    frame #2: 0x000000010000dcaf X4MeshTest`test_placeholder() at X4MeshTest.cc:27
    frame #3: 0x000000010000deb3 X4MeshTest`main(argc=1, argv=0x00007ffeefbfea98) at X4MeshTest.cc:39
    frame #4: 0x00007fff533b2015 libdyld.dylib`start + 1
    frame #5: 0x00007fff533b2015 libdyld.dylib`start + 1
(lldb) 

*/


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_convert_save(); 
    test_placeholder();


    return 0 ; 
}
