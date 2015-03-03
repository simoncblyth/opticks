#ifndef GGEO_H
#define GGEO_H

#include <vector>
class GMesh ; 
class GSolid ; 
class GMaterial ; 
class GSkinSurface ; 
class GBorderSurface ; 

class GGeo {
   public:
      GGeo();
      virtual ~GGeo();

   public:
      void add(GMesh*    mesh);
      void add(GSolid*    solid);
      void add(GMaterial* material);
      void add(GSkinSurface*  surface);
      void add(GBorderSurface*  surface);

   public:
      void Summary(const char* msg="GGeo::Summary");

   private:
      std::vector<GMesh*>    m_meshes ; 
      std::vector<GSolid*>    m_solids ; 
      std::vector<GMaterial*> m_materials ; 
      std::vector<GSkinSurface*>  m_skin_surfaces ; 
      std::vector<GBorderSurface*>  m_border_surfaces ; 

};

#endif


