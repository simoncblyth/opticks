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

#include "NGLM.hpp"
#include "NGrid3.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


typedef NGrid<glm::vec3, glm::ivec3, 3> G3 ; 
typedef NMultiGrid3<glm::vec3, glm::ivec3> MG3 ; 


void test_basics()
{
    int msk = (1 << 6) - 1 ;  // 0b111111 ;  

    G3 grid(3);

    LOG(info) << grid.desc();
    for(int loc=0 ; loc < grid.nloc ; loc++)
    {
        glm::ivec3 ijk = grid.ijk(loc) ;   // z-order morton index -> ijk
        int loc2 = grid.loc(ijk);      // ijk -> morton
        assert( loc2 == loc);

        glm::vec3  xyz = grid.fpos(ijk);    // ijk -> fractional 
        int loc3 = grid.loc(xyz);       // fractional -> morton
        assert( loc3 == loc);

        glm::ivec3 ijk2 = grid.ijk(xyz);   // fractional -> ijk 
        assert( ijk2 == ijk );

        if((loc & msk) == msk )
        {

             LOG(info) 
                   << " loc " << std::setw(5) << loc 
                   << " ijk " << glm::to_string(ijk)
                   << " fpos " << glm::to_string(xyz) 
                   ; 

        }

    }

}


void test_coarse_nominal()
{

    G3 nominal(7);
    G3 coarse(5);
    int elevation = nominal.level - coarse.level ;

    int n_loc = 0.45632*nominal.nloc ;    // some random loc

    glm::ivec3 n_ijk = nominal.ijk(n_loc) ; 
    glm::vec3  n_xyz = nominal.fpos(n_ijk) ;


    int msk = (1 << (elevation*3)) - 1 ;  
    int n_loc_0 = n_loc & ~msk ;   // nominal loc, but with low bits scrubbed -> bottom left of tile 

    glm::ivec3 n_ijk_0 = nominal.ijk(n_loc_0 ) ;  
    glm::vec3  n_xyz_0 = nominal.fpos(n_ijk_0 );
    
    // coarse level is parent or grandparent etc.. in tree
    // less nloc when go coarse : so must down shift 

    int c_loc = n_loc >> (3*elevation) ;
    int c_size = 1 << elevation ; 


    int c2n_loc = ( n_loc >> (3*elevation) ) << (3*elevation) ; // down and then back up
    assert(c2n_loc == n_loc_0);  // same as scrubbing the low bits


    glm::ivec3 c_ijk = coarse.ijk(c_loc) ; 
    glm::vec3  c_xyz = coarse.fpos(c_ijk );  

    glm::ivec3 c2n_ijk( c_ijk.x*c_size , c_ijk.y*c_size, c_ijk.z*c_size );


    std::cout 
           << " n_loc   " << std::setw(6) << n_loc
           << " n_ijk   " << glm::to_string(n_ijk)
           << " n_xyz   " << glm::to_string(n_xyz)
           << " (nominal) " 
           << std::endl 
           ;

    std::cout 
           << " n_loc_0 " << std::setw(6) << n_loc_0
           << " n_ijk_0 " << glm::to_string(n_ijk_0)
           << " n_xyz_0 " << glm::to_string(n_xyz_0)
           << " (nominal) with high res bits scrubbed  " 
           << std::endl 
           ;

    std::cout 
           << " c_loc   " << std::setw(6) << c_loc 
           << " c_ijk   " << glm::to_string(c_ijk)
           << " c_xyz   " << glm::to_string(c_xyz)
           << " (coarse coordinates : a different ballpark   " 
           << std::endl 
           ;

    std::cout 
           << " c2n_ijk " << glm::to_string(c2n_ijk)
           << " (coarse coordinates scaled back to nominal)    " 
           << std::endl 
           ;
 

}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    test_basics();
    test_coarse_nominal();

    MG3 mg ; 
    G3* g5 = mg.grid[5] ; 
    G3* g7 = mg.grid[7] ; 

    std::cout << " g5 " << g5->desc() << std::endl ; 
    std::cout << " g7 " << g7->desc() << std::endl ; 

    glm::vec3 fpos(0.1f, 0.2f, 0.3f ); 
    mg.dump("NMultiGrid3 dump, eg pos", fpos);
    mg.dump("NMultiGrid3 dump");
   

    return 0 ; 
}
