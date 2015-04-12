#pragma once

class GGeo ; 
class GMergedMesh ; 
class GDrawable ; 

class Geometry {
    public:
         Geometry();
         void load(const char* envprefix);

         GGeo*        getGGeo();
         GMergedMesh* getGeo();
         GDrawable*   getDrawable();

    private:
         GGeo*        m_ggeo ;    
         GMergedMesh* m_geo ;
     

};
