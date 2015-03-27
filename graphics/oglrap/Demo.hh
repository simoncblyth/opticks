#ifndef DEMO_H 
#define DEMO_H

#include "GArray.hh"
#include "GDrawable.hh"
#include "GVector.hh"
class GMesh ;

class Demo : public GDrawable {
  private:
      static const float pvertex[] ;
      static const float pcolor[] ;
      static const unsigned int pindex[] ;

      static const gfloat3 gvertex[] ;
      static const gfloat3 gcolor[] ;
      static const guint3  gindex[] ;

      static const float pmatrix[] ;

  public:
      Demo();
      virtual ~Demo();

      GMesh* asMesh();

      unsigned int getNumElements();
      GBuffer* getVertices();
      GBuffer* getColors();
      GBuffer* getIndices();
      GBuffer* getModelToWorld();

  private:
      GArray<float>* m_vertices ; 
      GArray<float>* m_colors ;
      GArray<unsigned int>* m_indices ; 
      GArray<float>* m_model_to_world ; 


};      


#endif



