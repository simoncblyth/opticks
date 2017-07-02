
#include "NSceneConfig.hpp"

NSceneConfig::NSceneConfig(const char* cfg)  
    :
    BConfig(cfg),
    check_surf_containment(0),
    check_aabb_containment(0)
{
    addInt("check_surf_containment", &check_surf_containment );
    addInt("check_aabb_containment", &check_aabb_containment );

    parse();
}


