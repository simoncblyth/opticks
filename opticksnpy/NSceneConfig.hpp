#pragma once

#include "BConfig.hh"
#include "NPY_API_EXPORT.hh"

struct NPY_API NSceneConfig : BConfig
{
    NSceneConfig(const char* cfg);

    int check_surf_containment ; 
    int check_aabb_containment ; 
    int disable_instancing     ;   // useful whilst debugging geometry subsets 
};

