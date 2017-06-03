#include "PLOG.hh"
#include "NPY_LOG.hh"

#include "NOpenMeshBoundary.hpp"
#include "NOpenMesh.hpp"

#include "NSphere.hpp"
#include "NBox.hpp"

typedef NOpenMeshType T ; 
typedef T::VertexHandle VH ; 
typedef T::FaceHandle   FH ; 


void test_write()
{
    const char* path = "/tmp/test_write.off" ;
    LOG(info) << "test_write " << path  ; 

    int level = 0 ; 
    int verbosity = 3 ; 
    int ctrl = 0 ; 

    NOpenMesh<T>* mesh = NOpenMesh<T>::cube(level, verbosity, ctrl );
    mesh->write(path);
    std::cout << mesh->brief() << std::endl ; 
}



void test_hexpatch()
{
    LOG(info) << "test_hexpatch" ; 
    int level = 0 ; 
    int verbosity = 3 ; 
    int ctrl = 0 ; 

    NOpenMesh<T>* mesh = NOpenMesh<T>::hexpatch(level, verbosity, ctrl  );
    mesh->dump();
    std::cout << mesh->desc.desc() ;
}


void test_cube()
{
    LOG(info) << "test_cube" ; 
    int level = 0 ; 
    int verbosity = 3 ; 
    int ctrl = 0 ; 


    NOpenMesh<T>* mesh = NOpenMesh<T>::cube(level, verbosity, ctrl  );
    mesh->dump();
    std::cout << mesh->desc.desc() ;
}

void test_tetrahedron()
{
    LOG(info) << "test_tetrahedron" ; 
    int level = 0 ; 
    int verbosity = 3 ; 
    int ctrl = 0 ; 


    NOpenMesh<T>* mesh = NOpenMesh<T>::tetrahedron( level, verbosity, ctrl );

    mesh->dump();

    std::cout << mesh->desc.desc() ;
}





void test_sphere_parametric()
{
    LOG(info) << "test_sphere_parametric" ;
    int verbosity = 1 ; 

    nsphere sphere = make_sphere(0,0,0,10);
    for(int level=2 ; level < 7 ; level++)
    { 
        NOpenMesh<T> m(&sphere, level, verbosity) ;
        //m.dump("sphere");
    }
}

void test_box_parametric()
{
    LOG(info) << "test_box_parametric" ;
    int verbosity = 1 ; 

    nbox box = make_box(0,0,0,100);
    for(int level=1 ; level < 6 ; level++)
    { 
        NOpenMesh<T> m(&box, level, verbosity);
        //m.dump("box");
    }
}


void test_add_vertex()
{
    LOG(info) << "test_add_vertex" ; 

    typedef T::Point        P ; 
    typedef T::VertexHandle VH ; 

    int level = 4 ; 
    int verbosity = 1 ; 

    NOpenMesh<T> mesh(NULL, level, verbosity) ;

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

    float epsilon = 1e-5f ; 

    typedef T::Point        P ; 
    typedef T::VertexHandle VH ; 


    int level = 4 ; 
    int verbosity = 1 ; 

    NOpenMesh<T> mesh(NULL, level, verbosity) ;

    VH vh[8];
    VH vhd[8];
    bool added(false) ; 

    vh[0] = mesh.add_vertex_unique(P(-1, -1,  1), added, epsilon);
    vhd[0] = mesh.add_vertex_unique(P(-1, -1,  1), added, epsilon);
    assert(vhd[0] == vh[0]);
    
    vh[1] = mesh.add_vertex_unique(P( 1, -1,  1), added, epsilon);
    vhd[1] = mesh.add_vertex_unique(P( 1, -1,  1), added, epsilon);
    assert(vhd[1] == vh[1]);

    vh[2] = mesh.add_vertex_unique(P( 1,  1,  1), added, epsilon);
    vhd[2] = mesh.add_vertex_unique(P( 1,  1,  1), added, epsilon);
    assert(vhd[2] == vh[2]);

    vh[3] = mesh.add_vertex_unique(P(-1,  1,  1),added, epsilon);
    vhd[3] = mesh.add_vertex_unique(P(-1,  1,  1),added, epsilon);
    assert(vhd[3] == vh[3]);

    vh[4] = mesh.add_vertex_unique(P(-1, -1, -1),added, epsilon);
    vhd[4] = mesh.add_vertex_unique(P(-1, -1, -1),added, epsilon);
    assert(vhd[4] == vh[4]);

    vh[5] = mesh.add_vertex_unique(P( 1, -1, -1),added, epsilon);
    vhd[5] = mesh.add_vertex_unique(P( 1, -1, -1),added, epsilon);
    assert(vhd[5] == vh[5]);

    vh[6] = mesh.add_vertex_unique(P( 1,  1, -1),added, epsilon);
    vhd[6] = mesh.add_vertex_unique(P( 1,  1, -1),added, epsilon);
    assert(vhd[6] == vh[6]);

    vh[7] = mesh.add_vertex_unique(P(-1,  1, -1),added, epsilon);
    vhd[7] = mesh.add_vertex_unique(P(-1,  1, -1),added, epsilon);
    assert(vhd[7] == vh[7]);


    mesh.dump();
}




void test_point()
{
    LOG(info) << "test_point" ; 

    typedef T::Point  P ; 

    P pt[2];

    pt[0] = P(1,1,1) ;
    pt[1] = P(1,1,1) ;

    P a(1,1,1);
    P b(1,1,1);
    P c(1,1,1.0001);
    P d  = c - b ;

    std::cout << "a " << a << " len " << a.length() << std::endl ; 
    std::cout << "b " << b << " len " << b.length() << std::endl ; 
    assert(a == b);

    std::cout << "c " << c << " len " << c.length() << std::endl ; 
    std::cout << "d " << d << " len " << d.length() << std::endl ; 





    std::cout << "pt[0] " << pt[0] << std::endl ; 
    std::cout << "pt[1] " << pt[1] << std::endl ; 


    


}



/*
        2
       / \
      /   \
     /     \
    0-------1

 i 0 ii 1
 i 1 ii 2
 i 2 ii 0

*/

void test_topology()
{
    int i, ii, n(3) ; 
    for (i=0, ii=1; i<n; ++i, ++ii, ii%=n)
       std::cout 
           << " i " << i 
           << " ii " << ii
           << std::endl ;  
}
 
void test_add_face()
{
    typedef T::Point        P ; 
    typedef T::VertexHandle VH ; 

    int level = 4 ; 
    int verbosity = 1 ; 
    NOpenMesh<T> m(NULL, level, verbosity) ;

    VH v00 = m.mesh.add_vertex(P(0, 0, 0));
    VH v01 = m.mesh.add_vertex(P(0, 1, 0));
    VH v10 = m.mesh.add_vertex(P(1, 0, 0));
 
    m.mesh.add_face( v00, v10, v01 );
    m.dump();

/*
    m.mesh.add_face( v00, v01, v10 );

  1

   B 01   
     | .      
  |  |   .    
  V  |     .  
   A 00-------10 C
  0       -->     2
   

 vh     0 p 0 0 0 heh     5 fvh->tvh 0->2 fh    -1 bnd     1
 vh     1 p 0 1 0 heh     1 fvh->tvh 1->0 fh    -1 bnd     1
 vh     2 p 1 0 0 heh     3 fvh->tvh 2->1 fh    -1 bnd     1


    m.mesh.add_face( v00, v10, v01 );

  1

   C 01   
     | .      
  ^  |   .    
  |  |     .  
   A 00-------10 B
  0      <--     2
 

 vh     0 p 0 0 0 heh     5 fvh->tvh 0->1 fh    -1 bnd     1
 vh     1 p 0 1 0 heh     3 fvh->tvh 1->2 fh    -1 bnd     1
 vh     2 p 1 0 0 heh     1 fvh->tvh 2->0 fh    -1 bnd     1


*/


}



void test_add_two_face()
{
    typedef T::Point        P ; 
    typedef T::VertexHandle VH ; 
    typedef T::FaceHandle   FH ; 


    int level = 4 ; 
    int verbosity = 1 ; 
    NOpenMesh<T> m(NULL, level, verbosity) ;

    VH v00 = m.mesh.add_vertex(P(0, 0, 0));
    VH v01 = m.mesh.add_vertex(P(0, 1, 0));
    VH v10 = m.mesh.add_vertex(P(1, 0, 0));
    VH v11 = m.mesh.add_vertex(P(1, 1, 0));
 
    FH f0 = m.mesh.add_face( v11, v00, v10 );
    assert(m.mesh.is_valid_handle(f0));

    // NB must do the check prior to adding the 2nd face 
    //    to be in same situation

    assert(m.is_consistent_face_winding(v00,v11,v01) == true);
    assert(m.is_consistent_face_winding(v11,v01,v00) == true);
    assert(m.is_consistent_face_winding(v01,v00,v11) == true);

    assert(m.is_consistent_face_winding(v11,v00,v01) == false);
    assert(m.is_consistent_face_winding(v00,v01,v11) == false);
    assert(m.is_consistent_face_winding(v01,v11,v00) == false);

    FH f1 = m.mesh.add_face( v00, v11, v01 );  // ok
    assert(m.mesh.is_valid_handle(f1));

    //FH f1 = m.mesh.add_face( v00, v11, v01 );   // ok
    //FH f1 = m.mesh.add_face( v11, v01, v00 );   // ok
    //FH f1 = m.mesh.add_face( v01, v00, v11 );   // ok

    //FH f1 = m.mesh.add_face( v11, v00, v01 );   // <-- invalid "complex edge" 
    //FH f1 = m.mesh.add_face( v00, v01, v11 );   // <-- invalid "complex edge" 
    //FH f1 = m.mesh.add_face( v01, v11, v00 );   // <-- invalid "complex edge" 
    //
    // Notice that the common edge between the two faces 
    // in oppositely wound for the two faces, 
    //
    //    v11->v00 in first
    //    v00->v11 in second
    //
    // Doing this wrong yields an invalid face:
    // 
    //      PolyMeshT::add_face: complex edge

    m.dump();

/*
     01-----11
     |     . |
     |   .   |
     | .     |
     00-----10


2017-05-27 18:34:49.367 INFO  [3752609] [>::dump@34] NOpenMesh::dump  V 4 F 2 E 5 euler [(V - E + F)]  (expect 2) 1
2017-05-27 18:34:49.368 INFO  [3752609] [>::dump_vertices@72] NOpenMesh::dump_vertices
 vh     0 p 0 0 0 heh     9 fvh->tvh 0->1 fh    -1 bnd     1
 vh     1 p 0 1 0 heh     7 fvh->tvh 1->3 fh    -1 bnd     1
 vh     2 p 1 0 0 heh     3 fvh->tvh 2->0 fh    -1 bnd     1
 vh     3 p 1 1 0 heh     5 fvh->tvh 3->2 fh    -1 bnd     1
2017-05-27 18:34:49.368 INFO  [3752609] [>::dump_faces@130] NOpenMesh::dump_faces nface 2
 f    0 i   0 v   3 :   3   0   2                1.000 1.000 0.000                0.000 0.000 0.000                1.000 0.000 0.000 
 f    1 i   1 v   3 :   0   3   1                0.000 0.000 0.000                1.000 1.000 0.000                0.000 1.000 0.000 
delta:tests blyth$ 



*/

}
 


void test_manual_subdivide_face()
{
    LOG(info) << "test_manual_subdivide_face" ; 

    int level = 0 ; 
    int verbosity = 3 ; 
    nbox box = make_box(0,0,0, 100);

    NOpenMesh<T> m(&box, level, verbosity) ;

    assert( m.find_boundary_loops() == 0 ) ;
 
    FH fh = *m.mesh.faces_begin() ; 

    m.manual_subdivide_face(fh, NULL ); 

    std::cout << "after manual_subdivide_face " << m.brief() << std::endl ;   

    assert( m.find_boundary_loops() == 0 ) ;


}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    //test_write(); 
    //test_cube(); 
    //test_tetrahedron(); 

    test_hexpatch(); 

    //test_add_vertex();
    //test_point();
 

    //test_box_parametric(); 
    //test_sphere_parametric(); 

    //test_add_vertex_unique();

    //test_topology();
    //test_add_face();
    //test_add_two_face();
 
    //test_subdivide_face(); 

    return 0 ; 
}
  
  
  
