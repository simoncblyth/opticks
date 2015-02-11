#ifndef GSOLID_H
#define GSOLID_H

class GMesh ;
class GMaterial ;
class GSurface ;

//
//  hmm the difference between the models is focussed in here 
//
//   chroma.geometry.Solid is all about splaying things across all the triangles
//
//  relationship between how many materials for each mesh is up for grabs
//

class GSolid {
  public:
      GSolid( GMesh* mesh, GMaterial* material1, GMaterial* material2, GSurface* surface1, GSurface* surface2 );
      virtual ~GSolid();

  public: 
      void Summary(const char* msg="GSolid::Summary");
 
  private:
      GMesh* m_mesh ; 
      GMaterial* m_material1 ; 
      GMaterial* m_material2 ; 
      GSurface*  m_surface1 ; 
      GSurface*  m_surface2 ; 

};


#endif
