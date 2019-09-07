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
#include <iomanip>

#include "GLMFormat.hpp"
#include "Nuv.hpp"
#include "NCone.hpp"

#include "OPTICKS_LOG.hh"


void test_sdf()
{
    LOG(info) << "test_sdf" ; 

    float r1 = 4.f ; 
    float z1 = 0.f ;

    float r2 = 2.f ; 
    float z2 = 2.f ;

    ncone* cone = make_cone(r1,z1,r2,z2) ; 
    nnode* node = (nnode*)cone ;

    for(float v=10. ; v >= -10. ; v-=1.f )
        std::cout 
               << " v       " << std::setw(10) << v 
               << " x       " << std::setw(10) << (*node)(v, 0, 0) 
               << " y       " << std::setw(10) << (*node)(0, v, 0) 
               << " z       " << std::setw(10) << (*node)(0, 0, v) 
               << " x(z=1)  " << std::setw(10) << (*node)(v, 0, 1.f) 
               << " y(z=1)  " << std::setw(10) << (*node)(0, v, 1.f) 
               << " x(z=-1) " << std::setw(10) << (*node)(v, 0, -1.f) 
               << " y(z=-1) " << std::setw(10) << (*node)(0, v, -1.f) 
               << std::endl ;  


}


void test_parametric()
{
    LOG(info) << "test_parametric" ; 

    float r1 = 4.f ; 
    float z1 = 0.f ;
    float r2 = 2.f ; 
    float z2 = 2.f ;

    ncone* cone = make_cone(r1,z1,r2,z2) ; 
 
    unsigned nsurf = cone->par_nsurf();
    assert(nsurf == 3);

    unsigned nu = 5 ; 
    unsigned nv = 5 ; 
    unsigned prim_idx = 0 ; 

    for(unsigned s=0 ; s < nsurf ; s++)
    {
        std::cout << " surf : " << s << std::endl ; 

        for(unsigned u=0 ; u <= nu ; u++){
        for(unsigned v=0 ; v <= nv ; v++)
        {
            nuv uv = make_uv(s,u,v,nu,nv, prim_idx );

            glm::vec3 p = cone->par_pos_model(uv);

            std::cout 
                 << " s " << std::setw(3) << s  
                 << " u " << std::setw(3) << u  
                 << " v " << std::setw(3) << v
                 << " p " << glm::to_string(p)
                 << std::endl ;   
        }
        }
    }
}




void test_getSurfacePointsAll()
{
    float r1 = 4.f ; 
    float z1 = 0.f ;
    float r2 = 2.f ; 
    float z2 = 2.f ;

    ncone* cone = make_cone(r1,z1,r2,z2) ; 

    cone->verbosity = 3 ;  
    cone->pdump("make_cone(4,0,2,2)");

    unsigned level = 5 ;  // +---+---+
    int margin = 1 ;      // o---*---o
    unsigned prim_idx = 0 ; 

    cone->collectParPoints( prim_idx, level, margin, FRAME_LOCAL, cone->verbosity); 
    const std::vector<glm::vec3>& surf = cone->par_points  ; 

    LOG(info) << "test_getSurfacePointsAll"
              << " surf " << surf.size()
              ;

    for(unsigned i=0 ; i < surf.size() ; i++ )
    {
        glm::vec3 p = surf[i]; 
        float sd = (*cone)(p.x, p.y, p.z);

        std::cout << " p " << gpresent(p) 
                  << " sd " << sd
                  << " sd(sci) " << std::scientific << sd << std::fixed 
                  << std::endl
                  ; 
    }

}






int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_sdf();
    //test_parametric();
    test_getSurfacePointsAll();

    return 0 ; 
} 
