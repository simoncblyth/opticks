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

#include <cmath>
#include <cassert>
#include <cmath>
#include <cstring>

#include <boost/math/constants/constants.hpp>

#include "OpticksCSG.h"

#include "NPrism.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"


nprism::nprism(float apex_angle_degrees, float height_mm, float depth_mm, float fallback_mm)
{
    param.f.x = apex_angle_degrees  ;
    param.f.y = height_mm  ;
    param.f.z = depth_mm  ;
    param.f.w = fallback_mm  ;
}

nprism::nprism(const nquad& param_)
{
    param = param_ ;
}

float nprism::height()
{
    return param.f.y > 0.f ? param.f.y : param.f.w ; 
}
float nprism::depth()
{
    return param.f.z > 0.f ? param.f.z : param.f.w ; 
}
float nprism::hwidth()
{
    float pi = boost::math::constants::pi<float>() ;
    return height()*tan((pi/180.f)*param.f.x/2.0f) ;
}


void nprism::dump(const char* msg)
{
    param.dump(msg);
}

nbbox nprism::bbox()
{
    float h  = height();
    float hw = hwidth();
    float d  = depth();

    nbbox bb ;
    bb.min = {-hw,0.f,-d/2.f } ;
    bb.max = { hw,  h, d/2.f } ;

    return bb ; 
}


npart nprism::part()
{
    // hmm more dupe of hemi-pmt.cu/make_prism
    // but if could somehow make vector types appear 
    // the same could use same code with CUDA ?

    nbbox bb = bbox();

    npart p ; 
    p.zero();            
    p.setParam(param) ; 
    p.setTypeCode(CSG_PRISM); 
    p.setBBox(bb);

    return p ; 
}


