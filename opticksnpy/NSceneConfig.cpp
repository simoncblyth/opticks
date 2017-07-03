
#include "PLOG.hh"
#include "NSceneConfig.hpp"

NSceneConfig::NSceneConfig(const char* cfg)  
    :
    BConfig(cfg),
    check_surf_containment(0),
    check_aabb_containment(0)
{

    LOG(info) << "NSceneConfig::NSceneConfig"
              << " cfg " << ( cfg ? cfg : " NULL " )
              ;

    addInt("check_surf_containment", &check_surf_containment );
    addInt("check_aabb_containment", &check_aabb_containment );

    parse();
}


