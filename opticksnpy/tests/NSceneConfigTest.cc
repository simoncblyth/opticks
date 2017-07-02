
#include "PLOG.hh"
#include "NPY_LOG.hh"

#include "NSceneConfig.hpp"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    const char* gltfconfig = "check_surf_containment=3,other=1,check_aabb_containment=214" ; 

    NSceneConfig cfg(gltfconfig);
    cfg.dump();


    return 0 ; 
}
