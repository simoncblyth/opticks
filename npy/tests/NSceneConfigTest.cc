/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

// TEST=NSceneConfigTest om-t

#include "OPTICKS_LOG.hh"
#include "NSceneConfig.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //const char* gltfconfig = "check_surf_containment=3,check_aabb_containment=214,csg_bbox_poly=1" ; 
    const char* gltfconfig = "check_surf_containment=3,check_aabb_containment=214,parsurf_epsilon=-4" ; 

    NSceneConfig cfg(gltfconfig);
    cfg.dump();

    assert( cfg.parsurf_epsilon == -4 );
  
    //assert( cfg.get_parsurf_epsilon() == 1e-4 );
    float eps = cfg.get_parsurf_epsilon() ;
    std::cout << " eps " << std::scientific << eps << std::endl ; 



    return 0 ; 
}
