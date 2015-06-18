#pragma once

#include "stdlib.h"

class GGeo ; 
class GMergedMesh ; 
class GDrawable ; 
class GBoundaryLibMetadata ; 


class Lookup ; 
class Sensor ; 

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
         GBoundaryLibMetadata*  getMetadata();
         GDrawable*             getDrawable();

         Lookup*                getMaterialLookup();
         Sensor*                getSensor();

    private:
         GLoaderImpFunctionPtr     m_imp ;  
         GGeo*                     m_ggeo ;    
         GMergedMesh*              m_mergedmesh ;
         GBoundaryLibMetadata*     m_metadata ;

         Lookup*                   m_lookup ; 
         Sensor*                   m_sensor ; 
     

};





inline GLoader::GLoader() 
   :
   m_ggeo(NULL),
   m_mergedmesh(NULL),
   m_metadata(NULL),
   m_lookup(NULL),
   m_sensor(NULL)
{
}




// setImp : sets implementation that does the actual loading
// using a function pointer to the implementation 
// allows ggeo-/GLoader depending on all the implementations

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
inline Lookup* GLoader::getMaterialLookup()
{
    return m_lookup ; 
}
inline Sensor* GLoader::getSensor()
{
    return m_sensor ; 
}











