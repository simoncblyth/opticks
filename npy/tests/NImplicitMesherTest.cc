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
#include <glm/glm.hpp>

#include "NImplicitMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NSphere.hpp"
#include "NBox.hpp"
#include "NBBox.hpp"

#include "OPTICKS_LOG.hh"



struct sphere_functor 
{
   sphere_functor( float x, float y, float z, float r, bool negate)
       :   
       center(x,y,z),
       radius(r),
       negate(negate)
   {   
   }   

   float operator()( float x, float y, float z) const
   {   
       glm::vec3 p(x,y,z) ;
       float d = glm::distance( p, center );
       float v = d - radius ; 
       return negate ? -v : v  ;
   }   

   std::string desc();


   glm::vec3 center ; 
   float     radius ; 
   bool      negate ; 

};


NTrianglesNPY* test_sphere_node()
{
    nsphere* sph = make_sphere(0,0,0, 10) ;
    nbbox bb = sph->bbox();

    LOG(info) << "test_sphere_node bb:" << bb.desc() ;

    int resolution = 100 ; 
    int verbosity = 1 ; 
    float bb_scale = 1.01 ; 

    NImplicitMesher im(sph, resolution, verbosity, bb_scale);
    NTrianglesNPY* tris = im();
    assert(tris);

    return tris ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_sphere_node();
    //test_sphere_functor();

    return 0 ; 
}
