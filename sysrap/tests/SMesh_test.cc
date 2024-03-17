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
    static int Concatenate(); 
    static int main(); 
};


inline int SMesh_test::Load()
{
    std::cout << "[SMesh_test::Load " << std::endl ; 
    glm::tmat4x4<double> tr = stra<double>::Translate(0., 0., 1000., 1. ) ; 
    SMesh* mesh = SMesh::Load("$MESH_FOLD", &tr ); 
    std::cout << mesh->desc(); 
    std::cout << "]SMesh_test::Load " << std::endl ; 
    return 0 ; 
}

inline int SMesh_test::Concatenate()
{
    std::cout << "[SMesh_test::Concatenate" << std::endl ; 
    glm::tmat4x4<double> a_tr = stra<double>::Translate(0., 0.,  1000., 1. ) ; 
    glm::tmat4x4<double> b_tr = stra<double>::Translate(0., 0., -1000., 1. ) ; 
    const SMesh* a_mesh = SMesh::Load("$MESH_FOLD", &a_tr ); 
    const SMesh* b_mesh = SMesh::Load("$MESH_FOLD", &b_tr ); 

    std::vector<const SMesh*> submesh ; 
    submesh.push_back(a_mesh); 
    submesh.push_back(b_mesh); 

    SMesh* mesh = SMesh::Concatenate(submesh); 
    std::cout << mesh->desc(); 
    std::cout << "]SMesh_test::Concatenate" << std::endl ; 
    return 0 ; 
}

inline int SMesh_test::main()
{
    int rc = 0 ; 
    rc += SMesh_test::Load() ;
    rc += SMesh_test::Concatenate() ;
    return rc ; 
}

int main()
{
    return SMesh_test::main(); 
}
