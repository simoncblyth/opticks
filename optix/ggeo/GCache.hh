#pragma once
#include "stdlib.h"
#include <string>
#include <cstring>
#include <cassert>
#include <map>

#include <glm/glm.hpp>

class NLog ; 
class GColors ; 
class GFlags ; 
class GGeo ; 
class Types ; 
class Typ ; 

// this is turning into GGeoConfig rather than just GCache 
// TODO: handle logging here, for control from tests
class GCache {
    public:
         static GCache* getInstance();
    private:
         // singleton instance
         static GCache* g_instance ; 
    public:
         static const char* JUNO ; 
         static const char* DAYABAY ; 
         static const char* PREFERENCE_BASE  ;
    public:
         GCache(const char* envprefix, const char* logname="ggeoview.log", const char* loglevel="info");
         void configure(int argc, char** argv);
    private:
         void init();
         void readEnvironment();  
    public:
         void setGeocache(bool geocache=true);
         bool isGeocache();
         //void setColors(GColors* colors);
         GColors* getColors();
         GFlags*  getFlags();
         Types*   getTypes();
         Typ*     getTyp();

         void setInstanced(bool instanced=true);
         bool isInstanced();
    public:
         GGeo*    getGGeo();
         void     setGGeo(GGeo* ggeo);
    public:
         const char* getIdPath();
         const char* getIdFold();  // parent directory of idpath
         std::string getRelativePath(const char* path); 
         std::string getObjectPath(const char* name, unsigned int ridx);
         std::string getMergedMeshPath(unsigned int ridx);
         std::string getPmtPath(unsigned int index);
         std::string getPropertyLibDir(const char* name);
    public:
         std::string getPreferenceDir(const char* type);
         bool loadPreference(std::map<std::string, std::string>& mss, const char* type, const char* name);
         bool loadPreference(std::map<std::string, unsigned int>& msu, const char* type, const char* name);
    public:
         const char* getEnvPrefix();
         bool idPathContains(const char* s); 
         void Summary(const char* msg="GCache::Summary");
    public:
         const char* getPath();
         const char* getQuery();
         const char* getCtrl();
         const char* getMeshfix();

         const char* getMeshfixCfg();
         //
         // 4 comma delimited floats specifying criteria for faces to be deleted from the mesh
         //
         //   xyz : face barycenter alignment 
         //     w : dot face normal cuts 
         //
         glm::vec4 getMeshfixFacePairingCriteria();

    public:
         const char* getDetector();
         bool        isJuno();
         bool        isDayabay();

    private:
          const char* m_envprefix ; 
          const char* m_logname  ; 
          const char* m_loglevel  ; 
          NLog*       m_log ; 
          GColors*    m_colors ; 
          GFlags*     m_flags ; 
          Types*      m_types ;
          Typ*        m_typ ;
          GGeo*       m_ggeo ; 
    private:
          const char* m_geokey ;
          const char* m_path ;
          const char* m_query ;
          const char* m_ctrl ;
          const char* m_meshfix ;
          const char* m_meshfixcfg ;
    private:
          const char* m_idpath ;
          const char* m_idfold ;
          const char* m_digest ;
    private:
          bool        m_geocache ; 
          bool        m_dayabay ; 
          bool        m_juno ; 
          const char* m_detector ;
          bool        m_instanced ; 
        
};


inline GCache* GCache::getInstance()
{
   return g_instance ;  
}

inline GCache::GCache(const char* envprefix, const char* logname, const char* loglevel)
       :
       m_envprefix(strdup(envprefix)),
       m_logname(strdup(logname)),
       m_loglevel(strdup(loglevel)),
       m_log(NULL),
       m_colors(NULL),
       m_flags(NULL),
       m_types(NULL),
       m_typ(NULL),
       m_ggeo(NULL),
       m_geokey(NULL),
       m_path(NULL),
       m_query(NULL),
       m_ctrl(NULL),
       m_meshfix(NULL),
       m_meshfixcfg(NULL),
       m_idpath(NULL),
       m_idfold(NULL),
       m_digest(NULL),
       m_geocache(false),
       m_dayabay(false),
       m_juno(false),
       m_detector(NULL),
       m_instanced(true)
{
       init();
       assert(g_instance == NULL && "GCache::GCache only one instance is allowed");
       g_instance = this ; 
}

inline const char* GCache::getIdPath()
{
    return m_idpath ;
}
inline const char* GCache::getIdFold()
{
    return m_idfold ;
}

inline const char* GCache::getEnvPrefix()
{
    return m_envprefix ;
}
inline const char* GCache::getPath()
{
    return m_path ;
}
inline const char* GCache::getQuery()
{
    return m_query ;
}
inline const char* GCache::getCtrl()
{
    return m_ctrl ;
}
inline const char* GCache::getMeshfix()
{
    return m_meshfix ;
}
inline const char* GCache::getMeshfixCfg()
{
    return m_meshfixcfg ;
}




inline void GCache::setGGeo(GGeo* ggeo)
{
    m_ggeo = ggeo ; 
}
inline GGeo* GCache::getGGeo()
{
    return m_ggeo ;
}





inline void GCache::setGeocache(bool geocache)
{
    m_geocache = geocache ; 
}
inline bool GCache::isGeocache()
{
    return m_geocache ;
}


inline void GCache::setInstanced(bool instanced)
{
   m_instanced = instanced ;
}
inline bool GCache::isInstanced()
{
   return m_instanced ; 
}


inline const char* GCache::getDetector()
{
    return m_detector ;
}
inline bool GCache::isJuno()
{
   return m_juno ; 
}
inline bool GCache::isDayabay()
{
   return m_dayabay ; 
}




inline bool GCache::idPathContains(const char* s)
{
    std::string idp(m_idpath);
    std::string ss(s);
    return idp.find(ss) != std::string::npos ;
}




inline std::string GCache::getRelativePath(const char* path)
{
    if(strncmp(m_idpath, path, strlen(m_idpath)) == 0)
    {
        return path + strlen(m_idpath) + 1 ; 
    }
    else
    {
        return path ;  
    }
}
