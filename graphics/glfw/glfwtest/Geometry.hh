#ifndef GEOMETRY_H 
#define GEOMETRY_H

#include "stddef.h"
#include "IGeometry.hh"

class Geometry : public IGeometry {
  private:
      static const float pvertex[] ;
      static const float pcolor[] ;
      static const unsigned int pindex[] ;

  public:
      Geometry();
      virtual ~Geometry();
      void load(const char* path=NULL);

      unsigned int getNumElements();
      Array<float>* getVertices();
      Array<float>* getColors();
      Array<unsigned int>* getIndices();

  private:
      void load_defaults();

  private:
      Array<float>* m_vertices ; 
      Array<float>* m_colors ;
      Array<unsigned int>* m_indices ; 


};      


#endif



