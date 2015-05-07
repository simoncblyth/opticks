#pragma once

#include "stdlib.h"

class GGeo ; 
class GMergedMesh ; 
class GDrawable ; 
class GSubstanceLibMetadata ; 

class Geometry {
    public:
         Geometry();

         static const char* identityPath( const char* envprefix);
         const char* load(const char* envprefix, bool nogeocache=false);
         void Summary(const char* msg);

         GGeo*                  getGGeo();
         GMergedMesh*           getMergedMesh();
         GSubstanceLibMetadata* getMetadata();
         GDrawable*             getDrawable();

    private:
         GGeo*                  m_ggeo ;    
         GMergedMesh*           m_mergedmesh ;
         GSubstanceLibMetadata* m_metadata ;
     

};


inline GGeo* Geometry::getGGeo()
{
    return m_ggeo ; 
}
inline GMergedMesh* Geometry::getMergedMesh()
{
    return m_mergedmesh ; 
}
inline GDrawable* Geometry::getDrawable()
{
    return (GDrawable*)m_mergedmesh ; 
}
inline GSubstanceLibMetadata* Geometry::getMetadata()
{
    return m_metadata ; 
}


