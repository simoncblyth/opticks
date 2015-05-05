#pragma once

class GGeo ; 
class GMergedMesh ; 
class GDrawable ; 

class Geometry {
    public:
         Geometry();

         const char* identityPath( const char* envprefix);
         void load(const char* envprefix);
         void Summary(const char* msg);

         GGeo*        getGGeo();
         GMergedMesh* getMergedMesh();
         GDrawable*   getDrawable();

    private:
         GGeo*        m_ggeo ;    
         GMergedMesh* m_mergedmesh ;
     

};
