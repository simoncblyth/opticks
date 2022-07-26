#include <cassert>
#include <iostream>
#include "CheckGeo.hh"
#include "sframe.h"

int main(int argc, char** argv)
{
    CheckGeo geo ; 
    SGeo* sg = (SGeo*)&geo ; 

    unsigned num_meshes_geo = geo.getNumMeshes() ; 
    unsigned num_meshes_sgeo = sg->getNumMeshes() ; 

    std::cout 
        << " num_meshes_geo " << num_meshes_geo 
        << " num_meshes_sgeo " << num_meshes_sgeo 
        << std::endl 
        ;    

    assert( num_meshes_geo == num_meshes_sgeo ); 


     
    sframe fr ; 
    sg->getFrame(fr, 0); 

    std::cout << fr.desc() << std::endl ; 
 



    return 0 ; 
}  
