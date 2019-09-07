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
#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"

template <typename T> class NPY  ;

class BRng ; 

/**
NRngDiffuse
============

This is using a sampling approach 
similar to that used all over G4OpBoundaryProcess

Other potential approaches:

* http://www.rorydriscoll.com/2009/01/07/better-sampling/

**/


class NPY_API NRngDiffuse
{
    public:
         NRngDiffuse(unsigned seed, float ctmin, float ctmax);
         std::string desc() const ; 

         float diffuse( glm::vec4& v, int& trials, const glm::vec3& dir) ;
         void uniform_sphere(glm::vec4& u );


        NPY<float>* uniform_sphere_sample(unsigned n);
        NPY<float>* diffuse_sample(unsigned n, const glm::vec3& dir);

    private:
         const float m_pi  ; 
         unsigned    m_seed ; 
         BRng*       m_uazi ; 
         BRng*       m_upol ; 
         BRng*       m_unct ; 
    

};
