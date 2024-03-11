/**

::

    ~/o/sysrap/tests/SMesh_test.sh 

**/

#include <iostream>
#include "SMesh.h"

int main()
{
    SMesh* mesh = SMesh::Load("$MESH_FOLD"); 
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
