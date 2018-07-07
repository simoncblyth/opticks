#pragma once

#include <map>
#include <vector>
#include <string>

class NSensorList ; 

class Opticks ; 
class OpticksKey ; 
class OpticksQuery ; 
class OpticksColors ; 
class OpticksFlags ; 
class OpticksAttrSeq ;

class SLog ; 
class BEnv ; 

class Types ;
class Typ ;

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

#include "BOpticksResource.hh"

/**
OpticksResource
=================

Canonical m_resource instance is resident of Opticks
instanciated by BOpticks::init **prior to configuration**.

WHY PRIOR TO CONFIG ?

* more convenient to be after, then can feed in some values



TODO:

* the split between whats in BOpticksResource and OpticksResource 
  is kinda arbitrary and makes this a pain to follow... 

* make the split more logical eg between installation resources and 
  specific geometry resources 

* detector specifics need to come in from json 

* move to constituent instead of base class, move all down to brap ? 

* relying on a set of envvars is annoying, as that divides config
  between scripts and here 

* DONE : envprefix was never anything other than OPTICKS_ so get rid of the machinery 

* need a better way to feed in metadata thru the keyhole, for live running 
  (probably a json string passed from user code ?) eg for OPTICKS_QUERY_LIVE 



**/


class OKCORE_API OpticksResource : public BOpticksResource {
    private:
       static const char* EMPTY ; 
       static const char* G4LIVE ; 
       static const char* JUNO ; 
       static const char* DAYABAY ; 
       static const char* DPIB ; 
       static const char* OTHER ; 
    private:
       static const char* PREFERENCE_BASE  ;
    public:
       static const char* DEFAULT_GEOKEY ;
       static const char* DEFAULT_QUERY ;
       static const char* DEFAULT_QUERY_LIVE ;
       static const char* DEFAULT_CTRL ;
       static const char* DEFAULT_MESHFIX ;
       static const char* DEFAULT_MESHFIX_CFG ;
    public:
       static const char* DEFAULT_MATERIAL_DYB ;
       static const char* DEFAULT_MATERIAL_JUNO ;
       static const char* DEFAULT_MATERIAL_OTHER ;
    public:
       static const char* DEFAULT_MEDIUM_DYB ;
       static const char* DEFAULT_MEDIUM_JUNO ;
       static const char* DEFAULT_MEDIUM_OTHER ;
    public:
       static const char* SENSOR_SURFACE_DYB ;
       static const char* SENSOR_SURFACE_JUNO ;
       static const char* SENSOR_SURFACE_OTHER ;
    public:
       static const char* EXAMPLE_MATNAMES_DYB ;
       static const char* EXAMPLE_MATNAMES_JUNO ;
       static const char* EXAMPLE_MATNAMES_OTHER ;
 
    public:
       static bool existsFile(const char* path);
       static bool existsFile(const char* dir, const char* name);
       static bool existsDir(const char* path);
    public:
       OpticksResource(Opticks* opticks=NULL, const char* lastarg=NULL);
       bool isValid();
    private:
       void init();
       BEnv* readIniEnvironment(const std::string& relpath);
       void readG4Environment();
       void readOpticksEnvironment();
       void readEnvironment();
       void readMetadata();
       void identifyGeometry();
       void assignDetectorName();
       void assignDefaultMaterial();
       void setValid(bool valid);
    public:

       const char* getDetectorBase();  // eg /usr/local/opticks/opticksdata/export/DayaBay 
       const char* getMaterialMap();   // eg /usr/local/opticks/opticksdata/export/DayaBay/ChromaMaterialMap.json 
       const char* getDefaultMaterial();  // material shortname based on the assigned detector, used for machinery tests only 
       const char* getDefaultMedium();    // PMT medium material name 
       const char* getExampleMaterialNames();  // comma delimited list of short material names
       const char* getSensorSurface(); 
    public:
       std::string getRelativePath(const char* path); 
       std::string getRelativePath(const char* name, unsigned int ridx);
       std::string getObjectPath(const char* name, unsigned int ridx);
       std::string getDetectorPath(const char* name, unsigned int ridx);
       std::string getMergedMeshPath(unsigned int ridx);

       std::string getBasePath(const char* rel); // wildcard path relative to IdBase export dir 
       std::string getPmtPath(unsigned int index, bool relative=false);
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
       bool idNameContains(const char* s); 
    public:
    public:
       void Summary(const char* msg="OpticksResource::Summary");
       void Dump(const char* msg="OpticksResource::Dump");
       std::string desc() const ; 
    public:
       //const char* getQueryString();
       const char* getCtrl();
       bool hasCtrlKey(const char* key) const ;
    public:
       // used to communicate test geometry config from geometry loading to test event writing 
       // see GGeoTest::initCreateCSG 
       void        setTestCSGPath(const char* testcsgpath);
       const char* getTestCSGPath() const ; 
       void        setTestConfig(const char* testconfig);
       const char* getTestConfig() const ; 
    public:
       // split these off as cannot assume users can write into geocache
       void saveFlags(const char* dir);
       void saveTypes(const char* dir);
    public:
       OpticksQuery* getQuery();
       OpticksColors* getColors();
       OpticksFlags*  getFlags();
       OpticksAttrSeq* getFlagNames();
       std::map<unsigned int, std::string> getFlagNamesMap();
   public:
       bool isDetectorType(const char* type_);
       bool isResourceType(const char* type_);

       Types*         getTypes();
       Typ*           getTyp();
       NSensorList*   getSensorList();
    private:
       std::string makeSidecarPath(const char* path, const char* styp=".dae", const char* dtyp=".ini");
    public:
       const char* getMeshfix();
       const char* getMeshfixCfg();
       glm::vec4   getMeshfixFacePairingCriteria();
    public:
       const char* getDetector();
       const char* getDetectorName();
       bool        isG4Live();
       bool        isJuno();
       bool        isDayabay();
       bool        isPmtInBox();
       bool        isOther();
   private:
       SLog*       m_log ; 
       Opticks*    m_ok ; 
       const char* m_lastarg ; 

   private:
       OpticksQuery*  m_query ;
   private:
       // results of readEnvironment
       const char* m_geokey ;
       const char* m_ctrl ;
       const char* m_meshfix ;
       const char* m_meshfixcfg ;
   private:
       bool        m_valid ; 
   private:
       OpticksColors* m_colors ;
       OpticksFlags*  m_flags ;
       OpticksAttrSeq* m_flagnames ;
       Types*         m_types ;
       Typ*           m_typ ;
       BEnv*          m_g4env ; 
       BEnv*          m_okenv ; 
   private:
       // results of identifyGeometry
       bool        m_g4live ;
       bool        m_dayabay ; 
       bool        m_juno ; 
       bool        m_dpib ; 
       bool        m_other ; 
       const char* m_detector ;
       const char* m_detector_name ;
       const char* m_detector_base ;
       const char* m_resource_base ;
       const char* m_material_map  ;
       const char* m_default_material  ;
       const char* m_default_medium  ;
       const char* m_example_matnames  ;
       const char* m_sensor_surface  ;
   private:
       const char*  m_testcsgpath ;
       const char*  m_testconfig ;
       NSensorList* m_sensor_list ; 
       
   private:
       std::map<std::string, std::string> m_metadata ;  
       std::vector<std::string> m_detector_types ; 
       std::vector<std::string> m_resource_types ; 
};


#include "OKCORE_TAIL.hh"

