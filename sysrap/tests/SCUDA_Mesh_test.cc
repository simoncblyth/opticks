/**

~/o/sysrap/tests/SCUDA_Mesh_test.sh


optix7.5 p28
   Mixing build input types in a single geometry-AS is not allowed
   [will need separated triangulated GAS for guidetube]

**/


#include "scuda.h"
#include "SCUDA_Mesh.h"
#include "SOPTIX.h"
#include "SOPTIX_Mesh.h"

int main()
{
    SMesh* m = SMesh::Load("$SCENE_FOLD/scene/mesh_grup/3" ); 
    std::cout << m->desc() ; 

    SOPTIX ox ; 
    std::cout << ox.desc() << std::endl ; 

    SCUDA_Mesh* _mesh = new SCUDA_Mesh(m) ; 
    std::cout << _mesh->desc() ; 

    SOPTIX_Mesh* mesh = new SOPTIX_Mesh(&ox, _mesh) ;  
    std::cout << mesh->desc() ; 


    return 0 ; 
}
