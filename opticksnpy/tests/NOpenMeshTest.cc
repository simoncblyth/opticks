#include "PLOG.hh"
#include "NOpenMesh.hpp"
#include "NSphere.hpp"


void test_write()
{
    const char* path = "/tmp/test_write.off" ;
    LOG(info) << "test_write " << path  ; 

    NOpenMesh<NOpenMeshType> mesh;

    mesh.build_cube();
    mesh.write(path);

    std::cout << mesh.brief() << std::endl ; 
}


void test_dump()
{
    LOG(info) << "test_dump" ; 

    NOpenMesh<NOpenMeshType> mesh;

    mesh.build_cube();
    mesh.dump();
}


void test_build_parametric()
{
    LOG(info) << "test_build_parametric" ;
 
    nsphere s1 = make_sphere(0,0,3,10);
    nsphere s2 = make_sphere(0,0,1,10);

    NOpenMesh<NOpenMeshType> m1, m2 ;

    m1.build_parametric( &s1, 8, 8 );
    m2.build_parametric( &s2, 8, 8 );

    m1.dump("s1");
    m2.dump("s2");
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
 
    test_write(); 
    test_dump(); 
    test_build_parametric(); 
  
    return 0 ; 
}
  
  
  
