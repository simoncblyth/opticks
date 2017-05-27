#include "PLOG.hh"
#include "NPY_LOG.hh"
#include "NOpenMesh.hpp"

#include "NSphere.hpp"
#include "NBox.hpp"


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


void test_sphere_parametric()
{
    LOG(info) << "test_sphere_parametric" ;
 
    nsphere sphere = make_sphere(0,0,0,10);

    NOpenMesh<NOpenMeshType> m ;

    m.build_parametric( &sphere, 4, 4 );
    m.dump("sphere");
}

void test_box_parametric()
{
    LOG(info) << "test_box_parametric" ;
 
    nbox box = make_box(0,0,0,100);

    NOpenMesh<NOpenMeshType> m ;

    m.build_parametric( &box, 1, 1 );
    m.dump("box");
}


void test_add_vertex()
{
    LOG(info) << "test_add_vertex" ; 

    typedef NOpenMeshType   T ; 
    typedef T::Point        P ; 
    typedef T::VertexHandle VH ; 

    NOpenMesh<T> mesh ;

    VH vh[8];
    VH vhd[8];


    vh[0] = mesh.mesh.add_vertex(P(-1, -1,  1));
    mesh.dump("after vh[0]");

    vhd[0] = mesh.mesh.add_vertex(P(-1, -1,  1));
    mesh.dump("after vhd[0]");

    // nothing special happens with duplicate vertices, just gets added
    //  https://mailman.rwth-aachen.de/pipermail/openmesh/2011-August/000584.html

    vh[1] = mesh.mesh.add_vertex(P( 1, -1,  1));
    mesh.dump("after vh[1]");


    vh[2] = mesh.mesh.add_vertex(P( 1,  1,  1));
    vh[3] = mesh.mesh.add_vertex(P(-1,  1,  1));
    vh[4] = mesh.mesh.add_vertex(P(-1, -1, -1));
    vh[5] = mesh.mesh.add_vertex(P( 1, -1, -1));
    vh[6] = mesh.mesh.add_vertex(P( 1,  1, -1));
    vh[7] = mesh.mesh.add_vertex(P(-1,  1, -1));

}


void test_add_vertex_unique()
{
    LOG(info) << "test_add_vertex_unique" ; 

    typedef NOpenMeshType   T ; 
    typedef T::Point        P ; 
    typedef T::VertexHandle VH ; 

    NOpenMesh<T> mesh ;

    VH vh[8];
    VH vhd[8];

    vh[0] = mesh.add_vertex_unique(P(-1, -1,  1));
    vhd[0] = mesh.add_vertex_unique(P(-1, -1,  1));
    assert(vhd[0] == vh[0]);
    
    vh[1] = mesh.add_vertex_unique(P( 1, -1,  1));
    vhd[1] = mesh.add_vertex_unique(P( 1, -1,  1));
    assert(vhd[1] == vh[1]);

    vh[2] = mesh.add_vertex_unique(P( 1,  1,  1));
    vhd[2] = mesh.add_vertex_unique(P( 1,  1,  1));
    assert(vhd[2] == vh[2]);

    vh[3] = mesh.add_vertex_unique(P(-1,  1,  1));
    vhd[3] = mesh.add_vertex_unique(P(-1,  1,  1));
    assert(vhd[3] == vh[3]);

    vh[4] = mesh.add_vertex_unique(P(-1, -1, -1));
    vhd[4] = mesh.add_vertex_unique(P(-1, -1, -1));
    assert(vhd[4] == vh[4]);

    vh[5] = mesh.add_vertex_unique(P( 1, -1, -1));
    vhd[5] = mesh.add_vertex_unique(P( 1, -1, -1));
    assert(vhd[5] == vh[5]);

    vh[6] = mesh.add_vertex_unique(P( 1,  1, -1));
    vhd[6] = mesh.add_vertex_unique(P( 1,  1, -1));
    assert(vhd[6] == vh[6]);

    vh[7] = mesh.add_vertex_unique(P(-1,  1, -1));
    vhd[7] = mesh.add_vertex_unique(P(-1,  1, -1));
    assert(vhd[7] == vh[7]);


    mesh.dump();
}




void test_point()
{
    LOG(info) << "test_point" ; 

    typedef NOpenMeshType   T ; 
    typedef T::Point  P ; 

    P pt[2];

    pt[0] = P(1,1,1) ;
    pt[1] = P(1,1,1) ;

    P pta(1,1,1);
    P ptb(1,1,1);

    std::cout << "pta " << pta << std::endl ; 
    std::cout << "ptb " << ptb << std::endl ; 
    assert(pta == ptb);

    std::cout << "pt[0] " << pt[0] << std::endl ; 
    std::cout << "pt[1] " << pt[1] << std::endl ; 

}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    //test_write(); 
    //test_dump(); 
    //test_add_vertex();
    //test_point();
 

    //test_sphere_parametric(); 
    test_box_parametric(); 

    //test_add_vertex_unique();
 
    return 0 ; 
}
  
  
  
