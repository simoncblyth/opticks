#pragma once

#include <map>
#include <string>
#include <cstring>
#include <glm/glm.hpp>

class Opticks ; 
class OpticksQuery ; 
class OpticksColors ; 
class OpticksFlags ; 


class OpticksResource {
    private:
       static const char* JUNO ; 
       static const char* DAYABAY ; 
       static const char* DPIB ; 
       static const char* OTHER ; 
    private:
       static const char* PREFERENCE_BASE  ;
    public:
       static const char* DEFAULT_GEOKEY ;
       static const char* DEFAULT_QUERY ;
       static const char* DEFAULT_CTRL ;
    public:
       static bool existsFile(const char* path);
       static bool existsFile(const char* dir, const char* name);
       static bool existsDir(const char* path);
    public:
       OpticksResource(Opticks* opticks=NULL, const char* envprefix="OPTICKS_", const char* lastarg=NULL);
       bool isValid();
    private:
       void init();
       void readEnvironment();
       void readMetadata();
       void identifyGeometry();
       void setValid(bool valid);
    public:
       const char* getIdPath();
       const char* getIdFold();  // parent directory of idpath
       std::string getRelativePath(const char* path); 
       std::string getObjectPath(const char* name, unsigned int ridx, bool relative=false);
       std::string getMergedMeshPath(unsigned int ridx);
       std::string getPmtPath(unsigned int index, bool relative=false);
       std::string getPropertyLibDir(const char* name);
    public:
       std::string getPreferenceDir(const char* type, const char* udet=NULL, const char* subtype=NULL);
       bool loadPreference(std::map<std::string, std::string>& mss, const char* type, const char* name);
       bool loadPreference(std::map<std::string, unsigned int>& msu, const char* type, const char* name);
    public:
       bool loadMetadata(std::map<std::string, std::string>& mdd, const char* path);
       void dumpMetadata(std::map<std::string, std::string>& mdd);
       bool hasMetaKey(const char* key);
       const char* getMetaValue(const char* key);
    public:
       const char* getEnvPrefix();
       bool idPathContains(const char* s); 
       void Summary(const char* msg="OpticksResource::Summary");
       void Dump(const char* msg="OpticksResource::Dump");
    public:
       const char* getDAEPath();
       const char* getGDMLPath();
       const char* getQueryString();
       const char* getCtrl();
    public:
       OpticksQuery* getQuery();
       OpticksColors* getColors();
       OpticksFlags*  getFlags();
    private:
       std::string makeSidecarPath(const char* path, const char* styp=".dae", const char* dtyp=".ini");
    public:
       const char* getMetaPath();
    public:
       const char* getMeshfix();
       const char* getMeshfixCfg();
       glm::vec4   getMeshfixFacePairingCriteria();
    public:
       const char* getDetector();
       bool        isJuno();
       bool        isDayabay();
       bool        isPmtInBox();
       bool        isOther();
   private:
       Opticks*    m_opticks ; 
       const char* m_envprefix ; 
       const char* m_lastarg ; 
   private:
       // results of readEnvironment
       const char* m_geokey ;
       const char* m_daepath ;
       const char* m_gdmlpath ;
       const char* m_query_string ;
       const char* m_ctrl ;
       const char* m_metapath ;
       const char* m_meshfix ;
       const char* m_meshfixcfg ;
       const char* m_idpath ;
       const char* m_idfold ;
       const char* m_digest ;
       bool        m_valid ; 
   private:
       OpticksQuery*  m_query ;
       OpticksColors* m_colors ;
       OpticksFlags*  m_flags ;
   private:
       // results of identifyGeometry
       bool        m_dayabay ; 
       bool        m_juno ; 
       bool        m_dpib ; 
       bool        m_other ; 
       const char* m_detector ;
       
   private:
      std::map<std::string, std::string> m_metadata ;  
};


inline OpticksResource::OpticksResource(Opticks* opticks, const char* envprefix, const char* lastarg) 
    :
       m_opticks(opticks),
       m_envprefix(strdup(envprefix)),
       m_lastarg(lastarg ? strdup(lastarg) : NULL),
       m_geokey(NULL),
       m_daepath(NULL),
       m_gdmlpath(NULL),
       m_query_string(NULL),
       m_ctrl(NULL),
       m_metapath(NULL),
       m_meshfix(NULL),
       m_meshfixcfg(NULL),
       m_idpath(NULL),
       m_idfold(NULL),
       m_digest(NULL),
       m_valid(true),
       m_query(NULL),
       m_colors(NULL),
       m_flags(NULL),
       m_dayabay(false),
       m_juno(false),
       m_dpib(false),
       m_other(false),
       m_detector(NULL)
{
    init();
}


inline void OpticksResource::setValid(bool valid)
{
    m_valid = valid ; 
}
inline bool OpticksResource::isValid()
{
   return m_valid ; 
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
inline const char* OpticksResource::getDAEPath()
{
    return m_daepath ;
}
inline const char* OpticksResource::getGDMLPath()
{
    return m_gdmlpath ;
}
inline const char* OpticksResource::getMetaPath()
{
    return m_metapath ;
}


inline const char* OpticksResource::getQueryString()
{
    return m_query_string ;
}
inline OpticksQuery* OpticksResource::getQuery()
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
inline bool OpticksResource::isPmtInBox()
{
   return m_dpib ; 
}
inline bool OpticksResource::isOther()
{
   return m_other ; 
}



inline bool OpticksResource::idPathContains(const char* s)
{
    bool ret = false ; 
    if(m_idpath)
    {
        std::string idp(m_idpath);
        std::string ss(s);
        ret = idp.find(ss) != std::string::npos ;
    }
    return ret ; 
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
