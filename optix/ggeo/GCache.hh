#pragma once

#include <cstdlib>
#include <string>
#include <cstring>

class Opticks ; 
class OpticksResource ; 
class NLog ; 
class GColors ; 
class GFlags ; 
class GGeo ; 
class Types ; 
class Typ ; 


class GCache {
    public:
         static GCache* getInstance();
    private:
         // singleton instance
         static GCache* g_instance ; 
    public:
         GCache(const char* envprefix, const char* logname="ggeoview.log", const char* loglevel="info");
         void configure(int argc, char** argv);
         void Summary(const char* msg="GCache::Summary");
    private:
         void init(const char* envprefix, const char* logname, const char* loglevel);
    public:
         GColors* getColors();
         GFlags*  getFlags();
         Types*   getTypes();
         Typ*     getTyp();
         Opticks* getOpticks();
         OpticksResource* getResource();
    public:
         const char* getIdPath();
         const char* getIdFold();
         std::string getRelativePath(const char* path);
         const char* getLastArg();
         int getLastArgInt();

    //    bool isCompute();  // need to know this prior to standard configuration is done, so do in the pre-configure here

    public:
         GGeo*    getGGeo();
         void     setGGeo(GGeo* ggeo);

    private:
          Opticks*          m_opticks ; 
          OpticksResource*  m_resource; 
          GColors*          m_colors ; 
          GFlags*           m_flags ; 
          Types*            m_types ;
          Typ*              m_typ ;
          GGeo*             m_ggeo ; 

        
};


inline GCache* GCache::getInstance()
{
   return g_instance ;  
}

inline GCache::GCache(const char* envprefix, const char* logname, const char* loglevel)
       :
       m_opticks(NULL),
       m_resource(NULL),
       m_colors(NULL),
       m_flags(NULL),
       m_types(NULL),
       m_typ(NULL),
       m_ggeo(NULL)
{
       init(envprefix, logname, loglevel);
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



