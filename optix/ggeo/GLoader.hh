#pragma once

#include "stdlib.h"

class GCache ; 
class GGeo ; 
class GMergedMesh ; 
class GDrawable ; 
class GBoundaryLibMetadata ; 
class GMaterialIndex ; 
class GSurfaceIndex ; 
class GColors ; 

class Lookup ; 

class GLoader {
     public:
         typedef GGeo* (*GLoaderImpFunctionPtr)(const char* );

    public:
         GLoader();
         void setCache(GCache* cache);
         void setImp(GLoaderImpFunctionPtr imp);
         void load(bool nogeocache=false);

    public:
         //static const char* identityPath( const char* envprefix);
         void Summary(const char* msg);

    public:
         GGeo*                  getGGeo();
         GMergedMesh*           getMergedMesh();
         GBoundaryLibMetadata*  getMetadata();
         GDrawable*             getDrawable();
         GMaterialIndex*        getMaterials();
         GSurfaceIndex*         getSurfaces();
         GColors*               getColors();

         Lookup*                getMaterialLookup();

    private:
         GCache*                   m_cache ; 
         GLoaderImpFunctionPtr     m_imp ;  
         GGeo*                     m_ggeo ;    
         GMergedMesh*              m_mergedmesh ;
         GBoundaryLibMetadata*     m_metadata ;
         GMaterialIndex*           m_materials ;
         GSurfaceIndex*            m_surfaces ;
         GColors*                  m_colors ;  
         Lookup*                   m_lookup ; 
     
};

inline GLoader::GLoader() 
   :
   m_cache(NULL),
   m_ggeo(NULL),
   m_mergedmesh(NULL),
   m_metadata(NULL),
   m_materials(NULL),
   m_surfaces(NULL),
   m_colors(NULL),
   m_lookup(NULL)
{
}




// setImp : sets implementation that does the actual loading
// using a function pointer to the implementation 
// avoids ggeo-/GLoader depending on all the implementations

inline void GLoader::setImp(GLoaderImpFunctionPtr imp)
{
    m_imp = imp ; 
}

inline void GLoader::setCache(GCache* cache)
{
    m_cache = cache ; 
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

inline GColors* GLoader::getColors()
{
    return m_colors ; 
}

inline GMaterialIndex* GLoader::getMaterials()
{
    return m_materials ; 
}
inline GSurfaceIndex* GLoader::getSurfaces()
{
    return m_surfaces ; 
}






