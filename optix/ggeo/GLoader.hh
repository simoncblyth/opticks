#pragma once

#include "stdlib.h"

class GGeo ; 
class GMergedMesh ; 
class GDrawable ; 
class GBoundaryLibMetadata ; 
//class GLoaderImp ; 

class GLoader {
     public:
         typedef GGeo* (*GLoaderImpFunctionPtr)(const char* );

    public:
         GLoader();
         void setImp(GLoaderImpFunctionPtr imp);

         static const char* identityPath( const char* envprefix);
         const char* load(const char* envprefix, bool nogeocache=false);
         void Summary(const char* msg);

         GGeo*                  getGGeo();
         GMergedMesh*           getMergedMesh();
         GBoundaryLibMetadata* getMetadata();
         GDrawable*             getDrawable();

    private:
         GLoaderImpFunctionPtr     m_imp ;  
         GGeo*                     m_ggeo ;    
         GMergedMesh*              m_mergedmesh ;
         GBoundaryLibMetadata*    m_metadata ;
     

};


inline void GLoader::setImp(GLoaderImpFunctionPtr imp)
{
    m_imp = imp ; 
}

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
inline GBoundaryLibMetadata* GLoader::getMetadata()
{
    return m_metadata ; 
}


