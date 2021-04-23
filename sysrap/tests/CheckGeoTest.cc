#include <cassert>
#include <iostream>
#include "CheckGeo.hh"

int main(int argc, char** argv)
{
    CheckGeo geo ; 
    SGeo* sgeo = (SGeo*)&geo ; 

    unsigned num_meshes_geo = geo.getNumMeshes() ; 
    unsigned num_meshes_sgeo = sgeo->getNumMeshes() ; 

    std::cout 
        << " num_meshes_geo " << num_meshes_geo 
        << " num_meshes_sgeo " << num_meshes_sgeo 
        << std::endl 
        ;    

    assert( num_meshes_geo == num_meshes_sgeo ); 

    return 0 ; 
}  
