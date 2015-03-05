#ifndef GSOLID_H
#define GSOLID_H

class GMesh ;
class GPropertyMap ;

#include "GMatrix.hh"


//  hmm the difference between the models is focussed in here 
//   chroma.geometry.Solid is all about splaying things across all the triangles
//  relationship between how many materials for each mesh is up for grabs
//

class GSolid {
  public:
      GSolid( GMatrixF* transform, GMesh* mesh, GPropertyMap* imaterial, GPropertyMap* omaterial, GPropertyMap* isurface, GPropertyMap* osurface );
      virtual ~GSolid();

  public:
     void setInnerMaterial(GPropertyMap* imaterial);
     void setOuterMaterial(GPropertyMap* omaterial);
     void setInnerSurface(GPropertyMap* isurface);
     void setOuterSurface(GPropertyMap* osurface);

  public:
     GPropertyMap* getInnerMaterial();
     GPropertyMap* getOuterMaterial();
     GPropertyMap* getInnerSurface();
     GPropertyMap* getOuterSurface();


  public: 
      void Summary(const char* msg="GSolid::Summary");
 
  private:
      GMatrixF* m_transform ; 
      GMesh*     m_mesh ; 
      GPropertyMap*  m_imaterial ; 
      GPropertyMap*  m_omaterial ; 
      GPropertyMap*  m_isurface ; 
      GPropertyMap*  m_osurface ; 

};


#endif
