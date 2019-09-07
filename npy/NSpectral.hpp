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

#include <vector>

#include "NGLM.hpp"
#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API NSpectral {
    public:
        typedef std::vector<unsigned int> VU_t ; 
        static const char* sRGB_D65 ;
    public:
        NSpectral(unsigned int num_colors=100, unsigned int wlmin=380, unsigned int wlmax=780);
    public:
        const std::vector<unsigned int>& getColorCodes();
        void dump(const char* msg="NSpectral::dump"); 
    private:
        void init();
        glm::vec3 getXYZ(float wavelength);
        unsigned int getColorCode(float wavelength);
    private:
        unsigned int m_num_colors ; 
        unsigned int m_wlmin ; 
        unsigned int m_wlmax ; 
        glm::mat3    m_XYZ2RGB ; 
        VU_t m_color_codes ; 
};

#include "NPY_TAIL.hh"


