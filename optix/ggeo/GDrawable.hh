#ifndef GDRAWABLE_H 
#define GDRAWABLE_H

#include "GBuffer.hh"
#include "GVector.hh"
#include <vector>

class GDrawable {
  public:
      virtual ~GDrawable(){}

      virtual GBuffer* getVerticesBuffer() = 0;
      virtual GBuffer* getNormalsBuffer() = 0;
      virtual GBuffer* getColorsBuffer() = 0;
      virtual GBuffer* getTexcoordsBuffer() = 0;
      virtual GBuffer* getIndicesBuffer() = 0;
      virtual GBuffer* getNodesBuffer() = 0;
      virtual GBuffer* getSubstancesBuffer() = 0;
      virtual GBuffer* getModelToWorldBuffer() = 0;
      virtual GBuffer* getWavelengthBuffer() = 0;
      virtual std::vector<unsigned int>& getDistinctSubstances() = 0;

      virtual gfloat4 getCenterExtent(unsigned int index) = 0 ;
      virtual unsigned int findContainer(gfloat3 p) = 0 ;

};      

#endif



