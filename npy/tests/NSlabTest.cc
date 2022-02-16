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

// TEST=NSlabTest om-t

#include <iostream>
#include <iomanip>

#include "SSys.hh"

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "NCSG.hpp"
#include "NSceneConfig.hpp"
#include "NNode.hpp"

#include "NSlab.hpp"
#include "NSphere.hpp"

#include "OPTICKS_LOG.hh"


void test_sdf()
{
    float near = -10 ; 
    float far  =  10 ; 
    float mid = (near+far)/2.f ; 

    nslab* s = make_slab( 0,0,1, near, far) ; 

    for(int i=-20 ; i < 20 ; i++)
        std::cout << std::setw(4) << i << " " << (*s)(0,0,i) << std::endl ;  

    assert( (*s)(0,0,near) == 0.f );
    assert( (*s)(0,0,far)  == 0.f );
    assert( (*s)(0,0,mid)  < 0.f );
    assert( (*s)(0,0,far+1) > 0.f );
    assert( (*s)(0,0,near-1) > 0.f );
}

void test_intersect()
{
    LOG(info) ; 

    float a = -10 ; 
    float b = 10 ; 

    nslab* s = make_slab( 0,0,1, a, b) ; 

    glm::vec3 ray_origin(0,0,0);
    glm::vec3 ray_direction(0,0,0);

    float tmin = 0.f ; 
    float t_expect ; 

    for(int i=0 ; i < 2 ; i++)
    {
        switch(i)
        {
            case 0: ray_direction.z = -1 ; t_expect = fabsf(a) ; break ;
            case 1: ray_direction.z = 1  ; t_expect = fabsf(b)  ; break ;
        }
        // intersect t is distance from ray origin to intersect
        // which is required to be greater than tmin eg 0., resulting in +ve t  
 
        glm::vec4 isect ; 
        bool has_intersect = s->intersect( tmin,  ray_origin, ray_direction, isect );
        float sd = (*s)(ray_origin.x, ray_origin.y, ray_origin.z);

        std::cout 
                  << " i " << i 
                  << " ray_origin "  << ray_origin
                  << " ray_direction " << ray_direction
                  << " has_intersect " << has_intersect 
                  << " sd(ray_origin) " << sd
                  << " isect " << isect 
                  << " t_expect " << t_expect 
                  << std::endl 
                  ;

        assert( has_intersect && isect.w == t_expect );
    }     

}




void test_slab_sphere_intersection()
{
    nsphere* a = nsphere::Create( 0.000,0.000,0.000,500.000 ) ; a->label = "a" ;   
    nslab* b = make_slab( 1.000,1.000,1.000,0.000,-500.000,100.000,0.000,0.000 ) ; b->label = "b" ;   
    nintersection* ab = nintersection::make_intersection( a, b ) ; ab->label = "ab" ; a->parent = ab ; b->parent = ab ;  ;   
    // hmm : why is the parenting not done within make_intersection ?

    
    ab->update_gtransforms();
    ab->verbosity = SSys::getenvint("VERBOSITY", 1) ; 
    ab->dump() ; 

    const char* boundary = "Rock//perfectAbsorbSurface/Vacuum" ;

    ab->set_boundary(boundary); 
    const char* gltfconfig = "" ;  
    const NSceneConfig* config = new NSceneConfig(gltfconfig);
    unsigned soIdx = 0 ; 
    unsigned lvIdx = 0 ; 
    NCSG* csg = NCSG::Adopt(ab, config, soIdx, lvIdx);
    csg->dump();
    csg->dump_surface_points("dsp", 20);

}


void test_slab_parsurf()
{
    nslab* s = make_slab( 1.000,0.000,0.000,0.000,-500.000,100.000,0.000,0.000 ) ; s->label = "s" ;   


    unsigned verbosity = 1 ; 
    unsigned level = 4 ; 
    int margin = 0 ; 
    unsigned prim_idx = 0 ;

    NNodeFrameType frty = FRAME_MODEL ; 

    s->collectParPoints(prim_idx, level, margin, frty, verbosity );    // using the above branch


    for(unsigned i=0 ; i < s->par_points.size() ; i++)
    { 
        std::cout << std::setw(5) << i 
                  << " ps " << gpresent(s->par_points[i])
                  << std::endl ; 
    }
    


}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_sdf();
    //test_intersect();

    test_slab_sphere_intersection();
    //test_slab_parsurf();

    return 0 ; 
}

