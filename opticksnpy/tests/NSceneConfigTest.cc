
#include "PLOG.hh"
#include "NPY_LOG.hh"

#include "NSceneConfig.hpp"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    //const char* gltfconfig = "check_surf_containment=3,other=1,check_aabb_containment=214,csg_bbox_poly=1" ; 
    const char* gltfconfig = "check_surf_containment=3,other=1,check_aabb_containment=214,parsurf_epsilon=-4" ; 

    NSceneConfig cfg(gltfconfig);
    cfg.dump();

    assert( cfg.parsurf_epsilon == -4 );
  
    //assert( cfg.get_parsurf_epsilon() == 1e-4 );
    float eps = cfg.get_parsurf_epsilon() ;
    std::cout << " eps " << std::scientific << eps << std::endl ; 



    return 0 ; 
}
