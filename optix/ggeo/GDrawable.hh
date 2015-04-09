#ifndef GDRAWABLE_H 
#define GDRAWABLE_H

#include "GBuffer.hh"

class GDrawable {
  public:
      virtual ~GDrawable(){}

      virtual GBuffer* getVerticesBuffer() = 0;
      virtual GBuffer* getNormalsBuffer() = 0;
      virtual GBuffer* getColorsBuffer() = 0;
      virtual GBuffer* getIndicesBuffer() = 0;
      virtual GBuffer* getModelToWorldBuffer() = 0;

};      

#endif



