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

#pragma once

#include <string>
#include <vector>
#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"
struct nnode ; 
struct ncone ; 

/**
NTreeJUNO
===========

Rationalization:

1. replaces (tubs-torus) with cone 
2. replaces vacuum cap intersect/subtract with ellipsoid z-cuts
   (in the case of LV 18 this simplifies the tree to a single primitive 
    z-cut ellipsoid)   

**/

struct NPY_API NTreeJUNO
{
    static nnode* Rationalize(nnode* a);

    NTreeJUNO(nnode* root_) ;

    nnode* root ; 

    glm::vec3 e_axes ; 
    glm::vec2 e_zcut ; 
    glm::mat4 e_trs_unscaled ; 

    ncone* cone ; 



    ncone* replacement_cone() ; 
    void rationalize();

    static nnode* create(int lv);  
    // lv:18,19,20,21  negated lv are rationalized 

    typedef std::vector<int> VI ;
    static const VI LVS ; 


};


