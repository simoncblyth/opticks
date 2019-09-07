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
#include "NDisc.hpp"


void test_dumpSurfacePointsAll()
{
    LOG(info) << "test_dumpSurfacePointsAll" ;
    ndisc* ds = make_disc();
    ds->dumpSurfacePointsAll("ds.dumpSurfacePointsAll", FRAME_LOCAL);
}


void test_parametric()
{
    LOG(info) << "test_parametric" ;

    float radius = 100.f ; 
    float z1    = -0.1f ; 
    float z2    =  0.1f ; 

    ndisc* ds = make_disc(radius,z1,z2); 

    // hmm need flexibility wrt par steps, only need one step for body ?

    unsigned nsurf = ds->par_nsurf();
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

            glm::vec3 p = ds->par_pos_model(uv);

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



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_parametric();
    test_dumpSurfacePointsAll();

    return 0 ; 
} 
