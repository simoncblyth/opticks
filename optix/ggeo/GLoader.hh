#pragma once

#include "stdlib.h"
#include "GVector.hh"

// npy-
class Types ; 

class GCache ; 
class GGeo ; 
class GMergedMesh ; 
class GDrawable ; 
class GBoundaryLib ; 
class GBoundaryLibMetadata ; 
class GItemIndex ; 
class GColors ; 
class GBuffer ; 
class GTreeCheck ; 



class Lookup ; 

class GLoader {
     public:
         typedef GGeo* (*GLoaderImpFunctionPtr)(const char*, const char*, const char* );

    public:
         GLoader();
         void setTypes(Types* types);
         void setCache(GCache* cache);
         void setRepeatIndex(int repeatidx);
         void setImp(GLoaderImpFunctionPtr imp);
         void load(bool nogeocache=false);

    public:
         void setMeshVersion(const char* mesh_version);
         void setInstanced(bool instanced=true);
         bool isInstanced();
    public:
         //static const char* identityPath( const char* envprefix);
         void Summary(const char* msg);

    public:
         GGeo*                  getGGeo();
         GMergedMesh*           getMergedMesh();
         GBoundaryLib*          getBoundaryLib();
         GBoundaryLibMetadata*  getMetadata();
         GDrawable*             getDrawable();
         GTreeCheck*            getTreeAnalyse();

         GItemIndex*            getMaterials();
         GItemIndex*            getSurfaces();
         GItemIndex*            getFlags();
         GItemIndex*            getMeshes();

         GColors*               getColors();
         Lookup*                getMaterialLookup();
         GBuffer*               getColorBuffer();
         gfloat4                getColorDomain();
         GBuffer*               getTransformsBuffer();
         int                    getRepeatIndex();

    private:
         Types*                    m_types ; 
         GCache*                   m_cache ; 
         GLoaderImpFunctionPtr     m_imp ;  
         GGeo*                     m_ggeo ;    
         GMergedMesh*              m_mergedmesh ;
         GBoundaryLib*             m_boundarylib ;
         GBoundaryLibMetadata*     m_metadata ;

         GItemIndex*               m_materials ;
         GItemIndex*               m_surfaces ;
         GItemIndex*               m_flags ;
         GItemIndex*               m_meshes ;

         GColors*                  m_colors ;  

         Lookup*                   m_lookup ; 

         GBuffer*                  m_color_buffer ; 
         gfloat4                   m_color_domain ; 

         GTreeCheck*               m_treeanalyse ;  
         int                       m_repeatidx ; 
         GBuffer*                  m_transforms_buffer ; 
         bool                      m_instanced ; 
         char*                     m_mesh_version ; 
};

inline GLoader::GLoader() 
   :
   m_types(NULL),
   m_cache(NULL),
   m_ggeo(NULL),
   m_mergedmesh(NULL),
   m_boundarylib(NULL),
   m_metadata(NULL),
   m_materials(NULL),
   m_surfaces(NULL),
   m_flags(NULL),
   m_meshes(NULL),
   m_colors(NULL),
   m_lookup(NULL),
   m_color_buffer(NULL),
   m_color_domain(0,0,0,0),
   m_treeanalyse(NULL),
   m_repeatidx(-1),
   m_transforms_buffer(NULL),
   m_instanced(true),
   m_mesh_version(NULL)
{
}

inline void GLoader::setMeshVersion(const char* mesh_version)
{
   m_mesh_version = strdup(mesh_version) ;
}




inline void GLoader::setInstanced(bool instanced)
{
   m_instanced = instanced ;
}

inline bool GLoader::isInstanced()
{
   return m_instanced ; 
}


// setImp : sets implementation that does the actual loading
// using a function pointer to the implementation 
// avoids ggeo-/GLoader depending on all the implementations

inline void GLoader::setImp(GLoaderImpFunctionPtr imp)
{
    m_imp = imp ; 
}
inline void GLoader::setTypes(Types* types)
{
    m_types = types ; 
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
inline GBoundaryLib* GLoader::getBoundaryLib()
{
    return m_boundarylib ; 
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

inline GItemIndex* GLoader::getMaterials()
{
    return m_materials ; 
}
inline GItemIndex* GLoader::getSurfaces()
{
    return m_surfaces ; 
}
inline GItemIndex* GLoader::getFlags()
{
    return m_flags ; 
}
inline GItemIndex* GLoader::getMeshes()
{
    return m_meshes ; 
}

inline GBuffer* GLoader::getColorBuffer()
{
    return m_color_buffer  ; 
}
inline gfloat4 GLoader::getColorDomain()
{
    return m_color_domain  ; 
}




inline int GLoader::getRepeatIndex()
{
    return m_repeatidx ; 
}
inline void GLoader::setRepeatIndex(int repeatidx)
{
    m_repeatidx = repeatidx  ; 
}



