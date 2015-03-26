#ifndef IGEOMETRY_H 
#define IGEOMETRY_H

#include "Array.hh"

class IGeometry {
  public:
      virtual ~IGeometry(){}

      virtual unsigned int getNumElements() = 0 ;
      virtual Array<float>* getVertices() = 0;
      virtual Array<float>* getColors() = 0;
      virtual Array<unsigned int>* getIndices() = 0;

};      

#endif



