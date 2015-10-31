#pragma once

#include "stdlib.h"
#include "GVector.hh"

class GCache ; 
class GGeo ; 
class GBoundaryLib ; 
class GBoundaryLibMetadata ; 
class GItemIndex ; 
class GColors ; 
class Lookup ; 

// THIS IS ON THE WAY OUT : TURNING IT INTO A HELPER FOR GGeo RATHER THAN A DRIVER
class GLoader {
     public:
    public:
         GLoader(GGeo* ggeo);
         void load(bool verbose=false);
    public:
         GBoundaryLibMetadata*  getMetadata();
         Lookup*                getMaterialLookup();
         GItemIndex*            getMaterials();
         GItemIndex*            getSurfaces();
         GItemIndex*            getFlags();
    private:
         GGeo*                     m_ggeo ;    
         GBoundaryLibMetadata*     m_metadata ;
         GItemIndex*               m_materials ;
         GItemIndex*               m_surfaces ;
         GItemIndex*               m_flags ;
         Lookup*                   m_lookup ; 
};

inline GLoader::GLoader(GGeo* ggeo) 
   :
   m_ggeo(ggeo),
   m_metadata(NULL),
   m_materials(NULL),
   m_surfaces(NULL),
   m_flags(NULL),
   m_lookup(NULL)
{
}

inline GBoundaryLibMetadata* GLoader::getMetadata()
{
    return m_metadata ; 
}
inline Lookup* GLoader::getMaterialLookup()
{
    return m_lookup ; 
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

