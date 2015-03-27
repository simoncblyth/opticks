#ifndef GDRAWABLE_H 
#define GDRAWABLE_H

#include "GBuffer.hh"

class GDrawable {
  public:
      virtual ~GDrawable(){}

      virtual unsigned int getNumFaces() = 0 ;
      virtual GBuffer* getVerticesBuffer() = 0;
      virtual GBuffer* getColorsBuffer() = 0;
      virtual GBuffer* getFacesBuffer() = 0;
      virtual GBuffer* getModelToWorldBuffer() = 0;

};      

#endif



