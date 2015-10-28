#pragma once

#include "stdlib.h"
#include "GVector.hh"

// npy-
class Types ; 

class GCache ; 
class GGeo ; 
class GMesh ; 
class GMergedMesh ; 
class GDrawable ; 
class GBoundaryLib ; 
class GBoundaryLibMetadata ; 
class GItemIndex ; 
class GColors ; 
class GBuffer ; 
class GTreeCheck ; 

class Lookup ; 


// THIS IS ON THE WAY OUT : TURNING IT INTO A HELPER FOR GGeo RATHER THAN A DRIVER
class GLoader {
     public:
    public:
         GLoader(GGeo* ggeo);
         void setTypes(Types* types);
         void setCache(GCache* cache);
         void setRepeatIndex(int repeatidx);
         void load(bool verbose=false);
    public:
         void setInstanced(bool instanced=true);
         bool isInstanced();

    public:
         GGeo*                  getGGeo();
         GMergedMesh*           getMergedMesh();
         GBoundaryLibMetadata*  getMetadata();
         GDrawable*             getDrawable();

         GItemIndex*            getMaterials();
         GItemIndex*            getSurfaces();
         GItemIndex*            getFlags();
         GItemIndex*            getMeshes();

         GColors*               getColors();
         Lookup*                getMaterialLookup();
         GBuffer*               getTransformsBuffer();
         int                    getRepeatIndex();

    private:
         GGeo*                     m_ggeo ;    
         Types*                    m_types ; 
         GCache*                   m_cache ; 
         GMergedMesh*              m_mergedmesh ;
         GBoundaryLibMetadata*     m_metadata ;

         GItemIndex*               m_materials ;
         GItemIndex*               m_surfaces ;
         GItemIndex*               m_flags ;
         GItemIndex*               m_meshes ;

         GColors*                  m_colors ;  

         Lookup*                   m_lookup ; 


         GTreeCheck*               m_treeanalyse ;  
         int                       m_repeatidx ; 
         GBuffer*                  m_transforms_buffer ; 
         bool                      m_instanced ; 
         char*                     m_mesh_version ; 
};

inline GLoader::GLoader(GGeo* ggeo) 
   :
   m_ggeo(ggeo),
   m_types(NULL),
   m_cache(NULL),
   m_mergedmesh(NULL),
   m_metadata(NULL),
   m_materials(NULL),
   m_surfaces(NULL),
   m_flags(NULL),
   m_meshes(NULL),
   m_colors(NULL),
   m_lookup(NULL),
   m_treeanalyse(NULL),
   m_repeatidx(-1),
   m_transforms_buffer(NULL),
   m_instanced(true),
   m_mesh_version(NULL)
{
}



inline void GLoader::setInstanced(bool instanced)
{
   m_instanced = instanced ;
}

inline bool GLoader::isInstanced()
{
   return m_instanced ; 
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


inline int GLoader::getRepeatIndex()
{
    return m_repeatidx ; 
}
inline void GLoader::setRepeatIndex(int repeatidx)
{
    m_repeatidx = repeatidx  ; 
}



