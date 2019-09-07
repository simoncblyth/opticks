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

#include "NPlane.hpp"

#include "GLMFormat.hpp"
#include "NGLMExt.hpp"

#include "OPTICKS_LOG.hh"



void test_sdf()
{
    float distToOrigin = 10 ; 

    nplane* plane = make_plane( 0,0,1, distToOrigin) ; 

    for(int i=0 ; i < 30 ; i++)
        std::cout << std::setw(4) << i << " " << (*plane)(0.f,0.f,i) << std::endl ;  
}

void test_intersect()
{
    nplane* plane = make_plane( 0,0,1,10) ;

    float tmin = 0.f ; 
    glm::vec3 ray_origin(0,0,0);
    glm::vec3 ray_direction(0,0,0);
 
    for(int i=0 ; i < 2 ; i++)
    {
        switch(i)
        {
           case 0: ray_direction.z = -1 ; break ;
           case 1: ray_direction.z =  1 ; break ;
        }

        glm::vec4 isect ; 
        bool valid_intersect = plane->intersect( tmin, ray_origin, ray_direction, isect );

        std::cout 
            <<  " i " << std::setw(2) << i 
            <<  " ray_origin   " << ray_origin
            <<  " ray_direction " << ray_direction
            <<  " isect " << isect
            <<  " valid_intersect " << valid_intersect 
            << std::endl 
            ; 
    }

}

void test_make_transformed()
{
    LOG(info) << "test_make_transformed" ; 

    nplane* pl = make_plane( 0,0,1,10) ;
    pl->pdump("pl");

    glm::mat4 r = nglmext::make_rotate(0,1,0,90); 
    glm::mat4 s = nglmext::make_scale(2,2,3); 
    glm::mat4 t = nglmext::make_translate(20,20,100); 

    std::vector<glm::mat4> tt ; 
    tt.push_back(r);
    tt.push_back(s);
    tt.push_back(t);

    for(unsigned i=0 ; i < tt.size() ; i++)
    {
        const glm::mat4& tr = tt[i] ;
        std::cout << gpresent("tr", tr) << std::endl  ; 
        glm::vec4 tpl = pl->make_transformed(tr);
        std::cout << gpresent("tpl", tpl) << std::endl  ; 
    }
}

void test_make_normal()
{
    glm::vec3 a(0,0,0) ;
    glm::vec3 b(1,0,0) ;
    glm::vec3 c(0,1,0) ;

    glm::vec3 x = make_normal(a,b,c);
    std::cout << gpresent("x", x ) << std::endl ; 
}


void test_make_plane3()
{
    glm::vec3 a(0,0,10) ;
    glm::vec3 b(1,0,10) ;
    glm::vec3 c(0,1,10) ;

    glm::vec4 pl = make_plane(a,b,c);
    std::cout << gpresent("pl", pl ) << std::endl ; 

}





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    //test_sdf();
    //test_intersect();

    //test_spawn_transformed();
    //test_make_normal();
    test_make_plane3();

    return 0 ; 
}
