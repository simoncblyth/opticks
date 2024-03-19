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
    static int Load0(); 
    static int Load(); 
    static int LoadTransformed();
    static int Concatenate(); 
    static int main(); 
};

inline int SMesh_test::Load0()
{
    const char* m = "0" ; 
    SMesh* mesh = SMesh::Load("$SCENE_FOLD/scene/mesh", m  ); 
    std::cout << mesh->desc(); 
    return 0 ;
}

inline int SMesh_test::Load()
{
    NPFold* scene_mesh = NPFold::Load("$SCENE_FOLD/scene/mesh"); 
    std::cout << scene_mesh->desc(); 

    int num_sub = scene_mesh->get_num_subfold(); 

    for(int i=0 ; i < num_sub ; i++)
    {
        NPFold* sub = scene_mesh->get_subfold(i);
        SMesh* mesh = SMesh::Import( sub );  
        std::cout << mesh->desc(); 
    }
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
