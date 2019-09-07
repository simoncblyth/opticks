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

#include <iostream>
#include <limits>
#include <algorithm>
#include <functional>


#include "NGLM.hpp"
#include "NPY.hpp"
#include "GLMFormat.hpp"

#include "NDualContouringSample.hpp"
#include "NTrianglesNPY.hpp"
#include "NSphere.hpp"
#include "NBox.hpp"

#include "OPTICKS_LOG.hh"


std::function<float(float,float,float)> Density_Func = NULL ;


void test_f_fff( std::function<float(float, float, float)> ff )
{
    LOG(info) << " test_f_fff(0,0,0) " << ff(0,0,0) ; 
}

void test_f_vec( std::function<float(const glm::vec3&)> f_vec, const glm::vec3& v)
{
    LOG(info) << " test_f_vec(v) " << f_vec(v) ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    nsphere* sph = make_sphere(0,0,0, 10) ;
    //nbox* box = make_box(5,5,5, 10) ;

    NDualContouringSample dcs ;
    NTrianglesNPY* tris = dcs(sph) ;
    LOG(info) << " num tris " << tris->getNumTriangles(); 

    return 0 ; 
}
