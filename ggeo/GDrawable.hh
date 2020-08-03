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

#pragma once

#include <vector>

#include "GBuffer.hh"
#include "GVector.hh"
#include "GBBox.hh"


template <typename T> class NPY ;

#include "GGEO_API_EXPORT.hh"

/**
GDrawable
===========

Purely virtual protocol base class.
See Renderer::setDrawable for how this gets used.

NB GBuffer still in use, these are slated for replacement with NPY 

**/


class GGEO_API GDrawable {
  public:
      virtual ~GDrawable(){}

      virtual unsigned getIndex() const = 0;
      virtual GBuffer* getVerticesBuffer() = 0;
      virtual GBuffer* getNormalsBuffer() = 0;
      virtual GBuffer* getColorsBuffer() = 0;
      virtual GBuffer* getTexcoordsBuffer() = 0;
      virtual GBuffer* getIndicesBuffer() = 0;
      virtual GBuffer* getNodesBuffer() = 0;
      virtual GBuffer* getTransformsBuffer() const  = 0;
      virtual GBuffer* getIdentityBuffer() const = 0;

      virtual  NSlice* getInstanceSlice() const = 0 ; 
      virtual  NSlice* getFaceSlice() = 0 ; 
  
      virtual NPY<float>*        getITransformsBuffer() const = 0;
      virtual NPY<unsigned int>* getInstancedIdentityBuffer() const = 0;

      virtual GBuffer* getBoundariesBuffer() = 0;
      virtual GBuffer* getModelToWorldBuffer() = 0;
      virtual std::vector<unsigned int>& getDistinctBoundaries() = 0;

      virtual gfloat4 getCenterExtent(unsigned int index) const = 0 ;
      virtual gbbox     getBBox(unsigned int index) const = 0 ;
      virtual unsigned int findContainer(gfloat3 p) = 0 ;

};      




