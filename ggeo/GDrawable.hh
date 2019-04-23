#pragma once

#include <vector>

#include "GBuffer.hh"
#include "GVector.hh"
#include "GBBox.hh"


template <typename T> class NPY ;

#include "GGEO_API_EXPORT.hh"

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




