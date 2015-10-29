#pragma once
#include "stdlib.h"
#include <string>
#include <cstring>
#include <cassert>
#include <glm/glm.hpp>


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
         static const char* PREFERENCE_DIR  ;
    public:
         GCache(const char* envprefix);
    public:
         void setGeocache(bool geocache=true);
         bool isGeocache();
    public:
         const char* getIdPath();
         std::string getRelativePath(const char* path); 
         std::string getMergedMeshPath(unsigned int ridx);
         std::string getPropertyLibDir(const char* name);
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
         const char* getPreferenceDir();
         bool        isJuno();
         bool        isDayabay();

    private:
          void init();
          void readEnvironment();  
    private:
          const char* m_envprefix ; 
    private:
          const char* m_geokey ;
          const char* m_path ;
          const char* m_query ;
          const char* m_ctrl ;
          const char* m_meshfix ;
          const char* m_meshfixcfg ;
    private:
          const char* m_idpath ;
          const char* m_digest ;
    private:
          bool        m_geocache ; 
          bool        m_dayabay ; 
          bool        m_juno ; 
          const char* m_detector ;
          const char* m_prefdir ;
};


inline GCache* GCache::getInstance()
{
   return g_instance ;  
}

inline GCache::GCache(const char* envprefix)
       :
       m_envprefix(strdup(envprefix)),
       m_geokey(NULL),
       m_path(NULL),
       m_query(NULL),
       m_ctrl(NULL),
       m_meshfix(NULL),
       m_meshfixcfg(NULL),
       m_idpath(NULL),
       m_digest(NULL),
       m_geocache(false),
       m_dayabay(false),
       m_juno(false),
       m_detector(NULL),
       m_prefdir(NULL)
{
       init();
       assert(g_instance == NULL && "GCache::GCache only one instance is allowed");
       g_instance = this ; 
}

inline const char* GCache::getIdPath()
{
    return m_idpath ;
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




inline void GCache::setGeocache(bool geocache)
{
    m_geocache = geocache ; 
}
inline bool GCache::isGeocache()
{
    return m_geocache ;
}

inline const char* GCache::getDetector()
{
    return m_detector ;
}
inline const char* GCache::getPreferenceDir()
{
    return m_prefdir ? m_prefdir : PREFERENCE_DIR  ;
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
