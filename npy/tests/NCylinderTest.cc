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

#include "OPTICKS_LOG.hh"
#include "Nuv.hpp"
#include "NCylinder.hpp"





void test_dumpSurfacePointsAll()
{
    LOG(info) << "test_dumpSurfacePointsAll" ;
    ncylinder* cy = ncylinder::Create();
    cy->dumpSurfacePointsAll("cy.dumpSurfacePointsAll", FRAME_LOCAL);
}



void test_parametric()
{
    LOG(info) << "test_parametric" ;

    float radius = 10. ; 
    float z1 = -5.f ; 
    float z2 = 15.f ; 
    ncylinder* cy = ncylinder::Create(radius,z1,z2); 

    unsigned nsurf = cy->par_nsurf();
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

            glm::vec3 p = cy->par_pos_model(uv);

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



void test_sdf()
{
    LOG(info) << "test_sdf" ; 
    enum { nc = 3 }; 
    ncylinder* c[nc] ; 

    float radius = 200 ;  
    float z1 = 0 ; 
    float z2 = 400 ; 

    c[0] = ncylinder::Create(radius,z1,z2) ;   
    c[1] = ncylinder::Create(radius,z1+100.f,z2+100.f) ;   
    c[2] = ncylinder::Create(radius,z1-100.f,z2-100.f) ;   

    for(int i=0 ; i < nc ; i++) c[i]->dump();

    for(float z=-500. ; z <= 500. ; z+=100 )
    {
        std::cout << " z " << std::setw(10) << z ;
        for(int i=0 ; i < nc ; i++) std::cout << std::setw(10) << (*c[i])(0,0,z) ;
        std::cout << std::endl ;  
    }

    for(float x=-500. ; x <= 500. ; x+=100 )
    {
        std::cout << " x " << std::setw(10) << x ;
        for(int i=0 ; i < nc ; i++) std::cout << std::setw(10) << (*c[i])(x,0,0) ;
        std::cout << std::endl ;  
    }
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_sdf();
    //test_parametric();

    test_dumpSurfacePointsAll();

    return 0 ; 
}   
