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

#include "PLOG.hh"


#include "NPlaneFromPoints.hpp"
#include "NOpenMeshBoundary.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMesh.hpp"
#include "NOpenMeshZipper.hpp"

#include "NOpenMeshType.hpp"
#include "NOpenMeshEnum.hpp"



template <typename T>
NOpenMeshZipper<T>::NOpenMeshZipper(
          const NOpenMesh<T>& lhs, 
          const NOpenMesh<T>& rhs
         )
   :
   lhs(lhs), 
   rhs(rhs)
{
    init();
}

template <typename T>
void NOpenMeshZipper<T>::init()
{
    LOG(info) << "NOpenMeshZipper::init"
              << " lhs " << lhs.brief()
              << " rhs " << rhs.brief()
              ;

    dump();
}



template <typename T>
void NOpenMeshZipper<T>::dump()
{
    unsigned n_lhs_inner = lhs.find.inner_loops.size() ;
    unsigned n_rhs_inner = rhs.find.inner_loops.size() ;
    unsigned n_lhs_outer = lhs.find.outer_loops.size() ;
    unsigned n_rhs_outer = rhs.find.outer_loops.size() ;

    std::cout 
         << " n_lhs_inner " << n_lhs_inner
         << " n_rhs_inner " << n_rhs_inner
         << " n_lhs_outer " << n_lhs_outer
         << " n_rhs_outer " << n_rhs_outer
         << std::endl 
         ; 

    for(unsigned i=0 ; i < n_lhs_inner ; i++)  
        dump_boundary( i, lhs.find.inner_loops[i], "lhs_inner" );

    for(unsigned i=0 ; i < n_lhs_outer ; i++)
        dump_boundary( i, lhs.find.outer_loops[i], "lhs_outer" );

    for(unsigned i=0 ; i < n_rhs_inner ; i++)  
        dump_boundary( i, rhs.find.inner_loops[i], "rhs_inner" );

    for(unsigned i=0 ; i < n_rhs_outer ; i++)  
        dump_boundary( i, rhs.find.outer_loops[i], "rhs_outer" );
}



template <typename T>
void NOpenMeshZipper<T>::dump_boundary(int index, const NOpenMeshBoundary<T>& loop, const char* msg)
{
    LOG(info) 
           << msg << " " 
           << std::setw(5) << index  
           << loop.desc()
            ; 

    std::cout << " loop.frontier plane " << loop.frontier.desc() << std::endl ; 
    loop.frontier.dump();

}








template struct NOpenMeshZipper<NOpenMeshType> ;

