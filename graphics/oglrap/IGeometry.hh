#ifndef IGEOMETRY_H 
#define IGEOMETRY_H

#include "Buffer.hh"

class IGeometry {
  public:
      virtual ~IGeometry(){}

      virtual unsigned int getNumElements() = 0 ;
      virtual Buffer* getVertices() = 0;
      virtual Buffer* getColors() = 0;
      virtual Buffer* getIndices() = 0;

};      

#endif



