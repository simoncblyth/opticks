#pragma once
#include "stdlib.h"
#include <string>

class GCache {
    public:
         GCache(const char* envprefix);
         const char* getIdPath();
         const char* getEnvPrefix();
         bool idPathContains(const char* s); 
         void Summary(const char* msg="GCache::Summary");

    private:
          void readEnvironment(const char* envprefix);  
    private:
          const char* m_envprefix ; 
          const char* m_geokey ;
          const char* m_path ;
          const char* m_query ;
          const char* m_ctrl ;
          const char* m_idpath ;
          const char* m_digest ;

};


inline GCache::GCache(const char* envprefix)
       :
       m_envprefix(NULL),
       m_geokey(NULL),
       m_path(NULL),
       m_query(NULL),
       m_ctrl(NULL),
       m_idpath(NULL),
       m_digest(NULL)
{
       readEnvironment(envprefix);
}
  

inline const char* GCache::getIdPath()
{
    return m_idpath ;
}
inline const char* GCache::getEnvPrefix()
{
    return m_envprefix ;
}


inline bool GCache::idPathContains(const char* s)
{
    std::string idp(m_idpath);
    std::string ss(s);
    return idp.find(ss) != std::string::npos ;
}

