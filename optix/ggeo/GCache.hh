#pragma once

#include <cstdlib>
#include <string>
#include <cstring>

// npy-
class NLog ; 


// optickscore-
class Opticks ; 
class OpticksResource ; 
class OpticksColors ; 
class OpticksQuery ; 

// ggeo-
class GFlags ; 
class GGeo ; 

class Types ; 
class Typ ; 

// GCache is being usurped by Opticks, OpticksResource
// but is used too widely to migrate all at once
// so eat away from the inside whilst keeping some 
// of the API working via passthrus
//
// SO, FOR NEW RESOURCES : DIRECTLY USE Opticks/OpticksResource
//

class GCache {
    public:
         static GCache* getInstance();
    private:
         // singleton instance
         static GCache* g_instance ; 
    public:
         GCache(Opticks* opticks);
    public:
         void Summary(const char* msg="GCache::Summary");
    private:
         void init();
    public:
         OpticksColors* getColors();
         GFlags*  getFlags();
         Types*   getTypes();
         Typ*     getTyp();
         Opticks* getOpticks();
         OpticksResource* getResource();
         OpticksQuery*    getQuery();

    public:
         const char* getDAEPath();
         const char* getGDMLPath();
         const char* getIdPath();
         const char* getIdFold();
         std::string getRelativePath(const char* path);
         const char* getLastArg();
         int getLastArgInt();

    public:
         GGeo*    getGGeo();
         void     setGGeo(GGeo* ggeo);

    private:
          Opticks*          m_opticks ; 
          OpticksResource*  m_resource; 
          GFlags*           m_flags ; 
          Types*            m_types ;
          Typ*              m_typ ;
          GGeo*             m_ggeo ; 

        
};


inline GCache* GCache::getInstance()
{
   return g_instance ;  
}

inline GCache::GCache(Opticks* opticks)
       :
       m_opticks(opticks),
       m_resource(NULL),
       m_flags(NULL),
       m_types(NULL),
       m_typ(NULL),
       m_ggeo(NULL)
{        
       init();
}



inline void GCache::setGGeo(GGeo* ggeo)
{
    m_ggeo = ggeo ; 
}
inline GGeo* GCache::getGGeo()
{
    return m_ggeo ;
}
inline OpticksResource* GCache::getResource()
{
    return m_resource ; 
}
inline Opticks* GCache::getOpticks()
{
    return m_opticks ; 
}



