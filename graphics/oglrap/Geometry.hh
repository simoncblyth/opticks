#ifndef GEOMETRY_H 
#define GEOMETRY_H

#include "stddef.h"
#include "Array.hh"
#include "IGeometry.hh"

class Geometry : public IGeometry {
  private:
      static const float pvertex[] ;
      static const float pcolor[] ;
      static const unsigned int pindex[] ;

  public:
      Geometry(const char* path=NULL);
      virtual ~Geometry();
      void load(const char* path);

      unsigned int getNumElements();
      Buffer* getVertices();
      Buffer* getColors();
      Buffer* getIndices();

  private:
      void load_defaults();

  private:
      Array<float>* m_vertices ; 
      Array<float>* m_colors ;
      Array<unsigned int>* m_indices ; 


};      


#endif



