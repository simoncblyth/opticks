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

//  clang NGrid3_StandaloneTest.cc  NGrid3.cpp -I$(glm-dir) -lc++ -L$(opticks-prefix)/lib -lNPY && DYLD_LIBRARY_PATH=$(opticks-prefix)/lib ./a.out && rm a.out

#include <iostream>
#include <iomanip>

#include "NGLM.hpp"
#include "NGrid3.hpp"


void test_glm_grid()
{
    NGrid<glm::vec3, glm::ivec3, 3> grid(3) ; 

    int msk = (1 << 6) - 1 ;  // 0b111111 ;  
    std::cout << grid.desc() << std::endl ; 

    for(int loc=0 ; loc < grid.nloc ; loc++)
    {
        glm::ivec3 ijk = grid.ijk(loc) ;
        int loc2 = grid.loc(ijk);
        assert( loc2 == loc );

        glm::vec3 xyz = grid.fpos(ijk) ;
        int loc3 = grid.loc(xyz);
        assert( loc3 == loc );

        glm::ivec3 ijk2 = grid.ijk(xyz) ;
        assert( ijk2 == ijk );

        if((loc & msk) == msk )
        std::cout << " loc " << std::setw(5) << loc 
                  << " ijk " << glm::to_string(ijk) 
                  << " xyz " << glm::to_string(xyz) 
                  << std::endl ; 
    }
}

void test_n_grid()
{
    NGrid<nvec3, nivec3, 3> grid(3) ; 

    int msk = (1 << 6) - 1 ;  // 0b111111 ;  
    std::cout << grid.desc() << std::endl ; 

    for(int loc=0 ; loc < grid.nloc ; loc++)
    {
        nivec3 ijk = grid.ijk(loc) ;
        int loc2 = grid.loc(ijk);
        assert( loc2 == loc );

        nvec3 xyz = grid.fpos(ijk) ;
        int loc3 = grid.loc(xyz);
        assert( loc3 == loc );

        nivec3 ijk2 = grid.ijk(xyz) ;
        assert( ijk2 == ijk );

        if((loc & msk) == msk )
        std::cout << " loc " << std::setw(5) << loc 
                  << " ijk " << ijk.desc() 
                  << " xyz " << xyz.desc() 
                  << std::endl ; 
    }
}





int main(int argc, char** argv)
{
    test_glm_grid();
    test_n_grid();

    return 0 ; 
}
