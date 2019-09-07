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


#include <limits>

#include "NQuad.hpp"
#include "NPY.hpp"
#include "NBufferSpec.hpp"
#include "YOGGeometry.hh"
#include "PLOG.hh"

namespace YOG {

Geometry::Geometry(int count_ )
    :
    count(count_)
{
}


void Geometry::make_triangle()
{
    /*

         (0,1)
          2
          |\
          | \
          |  \
          |   \
          |    \
          0-----1     
        (0,0)  (1,0)

    */


    assert( count == 3 );

    vtx = NPY<float>::make(3, 4) ; 
    vtx->zero();
    vtx->setQuad(0,0,0,    0.f, 0.f, 0.f, 1.f );
    vtx->setQuad(1,0,0,    1.f, 0.f, 0.f, 1.f );
    vtx->setQuad(2,0,0,    0.f, 1.f, 0.f, 1.f );

    vtx->minmax(vtx_minf, vtx_maxf );

    NBufferSpec vtx_spec = vtx->getBufferSpec(); 

    assert( vtx_spec.bufferByteLength == 3*4*4 + 5*16 ) ; 
    assert( vtx_spec.headerByteLength == 5*16 ) ; 


    glm::uvec4 iidx(0,1,2,0) ; 
    idx = NPY<unsigned>::make(3, 1) ;  // arrange items to have 1 element  
    idx->zero();
    //idx->setQuadU( iidx, 0,0,0 );

    //            i  j  k  l   value
    idx->setValue(0, 0, 0, 0,  0);
    idx->setValue(1, 0, 0, 0,  1);
    idx->setValue(2, 0, 0, 0,  2);
 
    idx->minmax(idx_min, idx_max );

   // hmm a deficiency with ygltf model, always expects vector<float> no matter the componentType
    idx_minf = std::vector<float>(idx_min.begin(), idx_min.end()) ; 
    idx_maxf = std::vector<float>(idx_max.begin(), idx_max.end()) ;  

 
    NBufferSpec idx_spec = idx->getBufferSpec(); 

    assert( idx_spec.bufferByteLength == 3*4 + 5*16 ) ; 
    assert( idx_spec.headerByteLength == 5*16 ) ; 
}

} // namespace
