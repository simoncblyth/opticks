/**

::

    ~/o/sysrap/tests/SMesh_test.sh 

**/

#include <iostream>
#include "SMesh.h"

int main()
{
    SMesh* mesh = SMesh::Load("$MESH_FOLD"); 
    std::cout << "mesh.name " << mesh->name << std::endl; 

    return 0 ; 
}
