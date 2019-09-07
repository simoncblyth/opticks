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

#include <glm/fwd.hpp>

template <typename T> class GDomain ;
template <typename T> class GAry ;

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GCIE {
   public:
        static const char* sRGB_D65 ;
   public:
        GCIE(float wmin, float wmax, float wstep);
        void dump(const char* msg="GCIE::dump");
   public:
        glm::vec3 getXYZ(float wavelength); 
        glm::vec3 getRGB(float wavelength); 

        float getInterpolatedX(float wavelength);
        float getInterpolatedY(float wavelength);
        float getInterpolatedZ(float wavelength);

        float getXMax();
        float getYMax();
        float getZMax();
   private:
        void init();
   private:
       glm::mat3       m_XYZ2RGB ; 
       GDomain<float>* m_domain ; 
       GAry<float>*    m_nm ; 
       GAry<float>*    m_X ; 
       GAry<float>*    m_Y ; 
       GAry<float>*    m_Z ; 

};

#include "GGEO_TAIL.hh"


