/**

::

    ~/o/sysrap/tests/SMesh_test.sh 

**/

#include <iostream>
#include "SMesh.h"

int main()
{
    glm::tmat4x4<double> tr = stra<double>::Translate(0., 0., 1000., 1. ) ; 
    SMesh* mesh = SMesh::Load("$MESH_FOLD", &tr ); 

    std::cout 
        << "mesh.name " << mesh->name
        << std::endl 
        << mesh->descFace() 
        << mesh->descTri() 
        << mesh->descVtx() 
        << mesh->descTriVtx() 
        << mesh->descFaceVtx() 
        << mesh->descVtxNrm() 
        ; 

    return 0 ; 
}
