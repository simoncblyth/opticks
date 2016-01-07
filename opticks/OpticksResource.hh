#pragma once

#include <map>
#include <string>
#include <cstring>
#include <glm/glm.hpp>


class OpticksResource {
    private:
       static const char* JUNO ; 
       static const char* DAYABAY ; 
       static const char* PREFERENCE_BASE  ;
    public:
       OpticksResource(const char* envprefix);
    private:
       void init();
       void readEnvironment();
    public:
       const char* getIdPath();
       const char* getIdFold();  // parent directory of idpath
       std::string getRelativePath(const char* path); 
       std::string getObjectPath(const char* name, unsigned int ridx, bool relative=false);
       std::string getMergedMeshPath(unsigned int ridx);
       std::string getPmtPath(unsigned int index, bool relative=false);
       std::string getPropertyLibDir(const char* name);
    public:
       std::string getPreferenceDir(const char* type, const char* udet=NULL);
       bool loadPreference(std::map<std::string, std::string>& mss, const char* type, const char* name);
       bool loadPreference(std::map<std::string, unsigned int>& msu, const char* type, const char* name);
    public:
       const char* getEnvPrefix();
       bool idPathContains(const char* s); 
       void Summary(const char* msg="OpticksResource::Summary");
    public:
       const char* getPath();
       const char* getQuery();
       const char* getCtrl();
    public:
       const char* getMeshfix();
       const char* getMeshfixCfg();
       glm::vec4   getMeshfixFacePairingCriteria();
    public:
       const char* getDetector();
       bool        isJuno();
       bool        isDayabay();
   private:
       const char* m_envprefix ; 
   private:
       // results of readEnvironment
       const char* m_geokey ;
       const char* m_path ;
       const char* m_query ;
       const char* m_ctrl ;
       const char* m_meshfix ;
       const char* m_meshfixcfg ;
       const char* m_idpath ;
       const char* m_idfold ;
       const char* m_digest ;
       bool        m_dayabay ; 
       bool        m_juno ; 
       const char* m_detector ;
};


inline OpticksResource::OpticksResource(const char* envprefix) 
    :
       m_envprefix(strdup(envprefix)),
       m_geokey(NULL),
       m_path(NULL),
       m_query(NULL),
       m_ctrl(NULL),
       m_meshfix(NULL),
       m_meshfixcfg(NULL),
       m_idpath(NULL),
       m_idfold(NULL),
       m_digest(NULL),
       m_dayabay(false),
       m_juno(false),
       m_detector(NULL)
{
    init();
}


inline const char* OpticksResource::getIdPath()
{
    return m_idpath ;
}
inline const char* OpticksResource::getIdFold()
{
    return m_idfold ;
}
inline const char* OpticksResource::getEnvPrefix()
{
    return m_envprefix ;
}
inline const char* OpticksResource::getPath()
{
    return m_path ;
}
inline const char* OpticksResource::getQuery()
{
    return m_query ;
}
inline const char* OpticksResource::getCtrl()
{
    return m_ctrl ;
}
inline const char* OpticksResource::getMeshfix()
{
    return m_meshfix ;
}
inline const char* OpticksResource::getMeshfixCfg()
{
    return m_meshfixcfg ;
}




inline const char* OpticksResource::getDetector()
{
    return m_detector ;
}
inline bool OpticksResource::isJuno()
{
   return m_juno ; 
}
inline bool OpticksResource::isDayabay()
{
   return m_dayabay ; 
}

inline bool OpticksResource::idPathContains(const char* s)
{
    std::string idp(m_idpath);
    std::string ss(s);
    return idp.find(ss) != std::string::npos ;
}

inline std::string OpticksResource::getRelativePath(const char* path)
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
