/**
SMesh_test.cc
==============

::

    ~/o/sysrap/tests/SMesh_test.sh 

**/

#include <iostream>
#include "SMesh.h"

struct SMesh_test
{
    static int Load(); 
    static int LoadTransformed();
    static int Concatenate(); 
    static int main(); 
};

inline int SMesh_test::Load()
{
    //const char* m = "0" ;   // SEGV
    //const char* m = "1" ;   // malloc incorrect checksum for freed object
    //const char* m = "2" ;   // SMesh::SmoothNormals FATAL NOT expected
    const char* m = "3" ;   // works ?
    //const char* m = "4" ;   // malloc incorrect checksum for freed object
    //const char* m = "5" ;   // malloc incorrect checksum for freed object
    //const char* m = "6" ;   // malloc incorrect checksum for freed object 
    //const char* m = "7" ;   // malloc incorrect checksum for freed object 
    //const char* m = "8" ;   // malloc incorrect checksum for freed object 

    SMesh* mesh = SMesh::Load("/tmp/SScene_test/scene/mesh", m  ); 

    std::cout << mesh->desc(); 
    return 0 ; 
}

inline int SMesh_test::LoadTransformed()
{
    std::cout << "[SMesh_test::LoadTransformed " << std::endl ; 
    glm::tmat4x4<double> tr = stra<double>::Translate(0., 0., 1000., 1. ) ; 
    SMesh* mesh = SMesh::LoadTransformed("$MESH_FOLD", &tr ); 
    std::cout << mesh->desc(); 
    std::cout << "]SMesh_test::LoadTransformed " << std::endl ; 
    return 0 ; 
}

inline int SMesh_test::Concatenate()
{
    std::cout << "[SMesh_test::Concatenate" << std::endl ; 
    glm::tmat4x4<double> a_tr = stra<double>::Translate(0., 0.,  1000., 1. ) ; 
    glm::tmat4x4<double> b_tr = stra<double>::Translate(0., 0., -1000., 1. ) ; 
    const SMesh* a_mesh = SMesh::LoadTransformed("$MESH_FOLD", &a_tr ); 
    const SMesh* b_mesh = SMesh::LoadTransformed("$MESH_FOLD", &b_tr ); 

    std::vector<const SMesh*> submesh ; 
    submesh.push_back(a_mesh); 
    submesh.push_back(b_mesh); 

    int ridx = 0 ; 
    SMesh* mesh = SMesh::Concatenate(submesh, ridx); 
    std::cout << mesh->desc(); 
    std::cout << "]SMesh_test::Concatenate" << std::endl ; 
    return 0 ; 
}

inline int SMesh_test::main()
{
    int rc = 0 ; 
    rc += SMesh_test::Load() ;
    //rc += SMesh_test::LoadTransformed() ;
    //rc += SMesh_test::Concatenate() ;
    return rc ; 
}

int main()
{
    return SMesh_test::main(); 
}
