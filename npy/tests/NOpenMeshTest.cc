#include "OPTICKS_LOG.hh"

#include "NParameters.hpp"
#include "NOpenMeshBoundary.hpp"
#include "NOpenMesh.hpp"

#include "NSphere.hpp"
#include "NBox.hpp"

typedef NOpenMeshType T ; 
typedef T::VertexHandle VH ; 
typedef T::FaceHandle   FH ; 

typedef NOpenMesh<T> MESH ; 

enum {
  CUBE = 6,
  TETRAHEDRON = 4,
  HEXPATCH = 666
};  


void test_write()
{
    const char* path = "/tmp/test_write.off" ;
    LOG(info) << "test_write " << path  ; 

    NParameters meta ; 
    meta.add<int>("ctrl", CUBE ); 

    MESH* mesh = MESH::Make( NULL, &meta, NULL );
    mesh->write(path);
    std::cout << mesh->brief() << std::endl ; 
}



void test_hexpatch()
{
    LOG(info) << "test_hexpatch" ; 

    NParameters meta ; 
    meta.add<int>("ctrl", HEXPATCH ); 

    MESH* mesh = MESH::Make( NULL, &meta, NULL );
    mesh->dump();

    std::cout << mesh->desc.desc() ;
}


void test_cube()
{
    LOG(info) << "test_cube" ; 

    NParameters meta ; 
    meta.add<int>("ctrl", CUBE ); 

    MESH* mesh = MESH::Make( NULL, &meta, NULL );

    mesh->dump();
    std::cout << mesh->desc.desc() ;
}

void test_tetrahedron()
{
    LOG(info) << "test_tetrahedron" ; 

    NParameters meta ; 
    meta.add<int>("ctrl", TETRAHEDRON ); 

    MESH* mesh = MESH::Make( NULL, &meta, NULL );

    mesh->dump();

    std::cout << mesh->desc.desc() ;
}


void test_sphere_parametric()
{
    LOG(info) << "test_sphere_parametric" ;

    nsphere* sphere = make_sphere(0,0,0,10);
    for(int level=2 ; level < 7 ; level++)
    { 
        NParameters meta ; 
        meta.add<int>("level", level ); 
        meta.add<int>("verbosity", 1 ); 

        MESH* m = MESH::Make(sphere, &meta, NULL ) ;
        m->dump("sphere");
    }
}

void test_box_parametric()
{
    LOG(info) << "test_box_parametric" ;

    nbox* box = make_box(0,0,0,100);
    for(int level=1 ; level < 6 ; level++)
    { 
        NParameters meta ; 
        meta.add<int>("level", level ); 
        meta.add<int>("verbosity", 1 ); 

        MESH* m = MESH::Make(box, &meta, NULL );
        m->dump("box");
    }
}


void test_add_vertex()
{
    LOG(info) << "test_add_vertex" ; 

    typedef T::Point        P ; 
    typedef T::VertexHandle VH ; 


    NParameters meta ; 
    meta.add<int>("level", 4 ); 
    meta.add<int>("verbosity", 1 ); 

    MESH* m = MESH::Make(NULL, &meta, NULL ) ;

    VH vh[8];
    VH vhd[8];


    vh[0] = m->mesh.add_vertex(P(-1, -1,  1));
    m->dump("after vh[0]");

    vhd[0] = m->mesh.add_vertex(P(-1, -1,  1));
    m->dump("after vhd[0]");

    // nothing special happens with duplicate vertices, just gets added
    //  https://mailman.rwth-aachen.de/pipermail/openm->2011-August/000584.html

    vh[1] = m->mesh.add_vertex(P( 1, -1,  1));
    m->dump("after vh[1]");


    vh[2] = m->mesh.add_vertex(P( 1,  1,  1));
    vh[3] = m->mesh.add_vertex(P(-1,  1,  1));
    vh[4] = m->mesh.add_vertex(P(-1, -1, -1));
    vh[5] = m->mesh.add_vertex(P( 1, -1, -1));
    vh[6] = m->mesh.add_vertex(P( 1,  1, -1));
    vh[7] = m->mesh.add_vertex(P(-1,  1, -1));

}


void test_add_vertex_unique()
{
    LOG(info) << "test_add_vertex_unique" ; 


    typedef T::Point        P ; 
    typedef T::VertexHandle VH ; 

    NParameters meta ; 
    meta.add<int>("level", 4 ); 
    meta.add<int>("verbosity", 1 ); 

    MESH* m = MESH::Make(NULL, &meta, NULL);

    VH vh[8];
    VH vhd[8];
    bool added(false) ; 

    vh[0] = m->build.add_vertex_unique(P(-1, -1,  1), added);
    vhd[0] = m->build.add_vertex_unique(P(-1, -1,  1), added);
    assert(vhd[0] == vh[0]);
    
    vh[1] = m->build.add_vertex_unique(P( 1, -1,  1), added);
    vhd[1] = m->build.add_vertex_unique(P( 1, -1,  1), added);
    assert(vhd[1] == vh[1]);

    vh[2] = m->build.add_vertex_unique(P( 1,  1,  1), added);
    vhd[2] = m->build.add_vertex_unique(P( 1,  1,  1), added);
    assert(vhd[2] == vh[2]);

    vh[3] = m->build.add_vertex_unique(P(-1,  1,  1),added);
    vhd[3] = m->build.add_vertex_unique(P(-1,  1,  1),added);
    assert(vhd[3] == vh[3]);

    vh[4] = m->build.add_vertex_unique(P(-1, -1, -1),added);
    vhd[4] = m->build.add_vertex_unique(P(-1, -1, -1),added);
    assert(vhd[4] == vh[4]);

    vh[5] = m->build.add_vertex_unique(P( 1, -1, -1),added);
    vhd[5] = m->build.add_vertex_unique(P( 1, -1, -1),added);
    assert(vhd[5] == vh[5]);

    vh[6] = m->build.add_vertex_unique(P( 1,  1, -1),added);
    vhd[6] = m->build.add_vertex_unique(P( 1,  1, -1),added);
    assert(vhd[6] == vh[6]);

    vh[7] = m->build.add_vertex_unique(P(-1,  1, -1),added);
    vhd[7] = m->build.add_vertex_unique(P(-1,  1, -1),added);
    assert(vhd[7] == vh[7]);


    m->dump();
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


    NParameters meta ; 
    meta.add<int>("level", 4 ); 
    meta.add<int>("verbosity", 1 ); 

    MESH* m = MESH::Make(NULL, &meta, NULL);


    VH v00 = m->mesh.add_vertex(P(0, 0, 0));
    VH v01 = m->mesh.add_vertex(P(0, 1, 0));
    VH v10 = m->mesh.add_vertex(P(1, 0, 0));
 
    m->mesh.add_face( v00, v10, v01 );
    m->dump();

/*
    m->mesh.add_face( v00, v01, v10 );

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



    NParameters meta ; 
    meta.add<int>("level", 4 ); 
    meta.add<int>("verbosity", 1 ); 

    MESH* m = MESH::Make(NULL, &meta, NULL);

    VH v00 = m->mesh.add_vertex(P(0, 0, 0));
    VH v01 = m->mesh.add_vertex(P(0, 1, 0));
    VH v10 = m->mesh.add_vertex(P(1, 0, 0));
    VH v11 = m->mesh.add_vertex(P(1, 1, 0));
 
    FH f0 = m->mesh.add_face( v11, v00, v10 );
    assert(m->mesh.is_valid_handle(f0));

    // NB must do the check prior to adding the 2nd face 
    //    to be in same situation

    assert(m->build.is_consistent_face_winding(v00,v11,v01) == true);
    assert(m->build.is_consistent_face_winding(v11,v01,v00) == true);
    assert(m->build.is_consistent_face_winding(v01,v00,v11) == true);

    assert(m->build.is_consistent_face_winding(v11,v00,v01) == false);
    assert(m->build.is_consistent_face_winding(v00,v01,v11) == false);
    assert(m->build.is_consistent_face_winding(v01,v11,v00) == false);

    FH f1 = m->mesh.add_face( v00, v11, v01 );  // ok
    assert(m->mesh.is_valid_handle(f1));

    //FH f1 = m->mesh.add_face( v00, v11, v01 );   // ok
    //FH f1 = m->mesh.add_face( v11, v01, v00 );   // ok
    //FH f1 = m->mesh.add_face( v01, v00, v11 );   // ok

    //FH f1 = m->mesh.add_face( v11, v00, v01 );   // <-- invalid "complex edge" 
    //FH f1 = m->mesh.add_face( v00, v01, v11 );   // <-- invalid "complex edge" 
    //FH f1 = m->mesh.add_face( v01, v11, v00 );   // <-- invalid "complex edge" 
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

    m->dump();

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


    nbox* box = make_box(0,0,0, 100);


    NParameters meta ; 
    meta.add<int>("level", 4 ); 
    meta.add<int>("verbosity", 1 ); 

    MESH* m = MESH::Make(box, &meta, NULL) ;

    assert( m->find.find_boundary_loops() == 0 ) ;
 
    m->subdiv.sqrt3_refine( FIND_ALL_FACE, -1 ); 

    std::cout << "after m->nual_subdivide_face " << m->brief() << std::endl ;   

    assert( m->find.find_boundary_loops() == 0 ) ;


}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

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
  
  
  
