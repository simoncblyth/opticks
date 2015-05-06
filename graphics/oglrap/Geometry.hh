#pragma once

#include "stdlib.h"

class GGeo ; 
class GMergedMesh ; 
class GDrawable ; 

class Geometry {
    public:
         Geometry();

         const char* load(const char* envprefix, bool nogeocache=false);
         void Summary(const char* msg);

         GGeo*        getGGeo();
         GMergedMesh* getMergedMesh();
         GDrawable*   getDrawable();

    private:
         const char* identityPath( const char* envprefix, const char* ext=NULL);

    private:
         GGeo*        m_ggeo ;    
         GMergedMesh* m_mergedmesh ;
     

};
