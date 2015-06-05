#pragma once

#include "stdlib.h"

class GGeo ; 
class GMergedMesh ; 
class GDrawable ; 
class GSubstanceLibMetadata ; 


// TODO: rename to GLoader? and move with GeometryTest->GLoaderTest into GGeo 
class GLoader {
    public:
         GLoader();

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


inline GGeo* GLoader::getGGeo()
{
    return m_ggeo ; 
}
inline GMergedMesh* GLoader::getMergedMesh()
{
    return m_mergedmesh ; 
}
inline GDrawable* GLoader::getDrawable()
{
    return (GDrawable*)m_mergedmesh ; 
}
inline GSubstanceLibMetadata* GLoader::getMetadata()
{
    return m_metadata ; 
}


