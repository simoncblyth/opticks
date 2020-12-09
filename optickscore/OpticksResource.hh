/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <map>
#include <vector>
#include <string>

class SLog ; 

class BOpticksKey ; 
class BOpticksResource ; 
class BEnv ; 

class Opticks ; 
#ifdef OLD_RESOURCE
class OpticksQuery ; 
#endif
class OpticksColors ; 
class OpticksFlags ; 
class OpticksAttrSeq ;


class Types ;
class Typ ;

#include "plog/Severity.h"
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"


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

* need a better way to feed in metadata thru the keyhole, for live running 
  (probably a json string passed from user code ?) eg for OPTICKS_QUERY_LIVE 


**/


class OKCORE_API OpticksResource 
{
    public:
       static void SetupG4Environment();
       static BEnv* ReadIniEnvironment(const std::string& relpath);
    public:
       static const plog::Severity LEVEL ;  
    public:
       static bool existsFile(const char* path);
       static bool existsFile(const char* dir, const char* name);
       static bool existsDir(const char* path);
    public:
       OpticksResource(Opticks* ok=NULL);
       bool isValid();
    private:
       void init();
       void readOpticksEnvironment();
       void readEnvironment();
#ifdef OLD_RESOURCE
       void readMetadata();
       void identifyGeometry();
       void assignDetectorName();
       void assignDefaultMaterial();
#endif

       void initRunResultsDir();
       void setValid(bool valid);
    public:

#ifdef OLD_RESOURCE
       const char* getDetectorBase();  // eg /usr/local/opticks/opticksdata/export/DayaBay 
       const char* getMaterialMap();   // eg /usr/local/opticks/opticksdata/export/DayaBay/ChromaMaterialMap.json 
       const char* getDefaultMaterial();  // material shortname based on the assigned detector, used for machinery tests only 
       const char* getDefaultMedium();    // PMT medium material name 
       const char* getExampleMaterialNames();  // comma delimited list of short material names
       const char* getSensorSurface(); 
#endif
       int         getDefaultFrame() const ; 

    public:
       const char* getRunResultsDir() const ;
    public:
       std::string formCacheRelativePath(const char* path) const ; 
    public:
       std::string getRelativePath(const char* name, unsigned int ridx) const ;
       std::string getRelativePath(const char* name) const ;
       std::string getObjectPath(const char* name, unsigned int ridx) const ;
       std::string getObjectPath(const char* name) const ;
       std::string getPropertyLibDir(const char* name) const ;
       const char* getIdPath() const ;

   
    public:
#ifdef OLD_RESOURCE
       std::string getDetectorPath(const char* name, unsigned int ridx);
       std::string getPmtPath(unsigned int index, bool relative=false);
#endif
       std::string getMergedMeshPath(unsigned int ridx);

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
    public:
    public:
       void Summary(const char* msg="OpticksResource::Summary");
       void Dump(const char* msg="OpticksResource::Dump");
    public:
       OpticksQuery* getQuery() const ;
#ifdef OLD_RESOURCE
       const char* getCtrl();
       bool hasCtrlKey(const char* key) const ;
#endif
    public:
       // split these off as cannot assume users can write into geocache
       void saveFlags(const char* dir);
       void saveTypes(const char* dir);
    public:
       OpticksColors* getColors();
       OpticksFlags*  getFlags() const ;
       OpticksAttrSeq* getFlagNames();
       std::map<unsigned int, std::string> getFlagNamesMap();

   private:
       OpticksColors* loadColorMapFromPrefs(); 
   public:
       bool isDetectorType(const char* type_);
       bool isResourceType(const char* type_);

       Types*         getTypes();
       Typ*           getTyp();
    private:
       std::string makeSidecarPath(const char* path, const char* styp=".dae", const char* dtyp=".ini");

#ifdef OLD_RESOURCE
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
#endif
   public: 
       BOpticksResource*  getRsc() const ;
       // via m_rsc
       const char* getTestCSGPath() const ;
       const char* getTestConfig() const ;
       void        setTestCSGPath(const char* path) ;     
       void        setTestConfig(const char* config) ; 
   private:
       SLog*             m_log ; 
       BOpticksResource* m_rsc ; 
       BOpticksKey*      m_key ; 
       Opticks*          m_ok ; 
       OpticksQuery*     m_query ;
   private:
#ifdef OLD_RESOURCE
       // results of readEnvironment
       const char* m_geokey ;
       const char* m_ctrl ;
       const char* m_meshfix ;
       const char* m_meshfixcfg ;
#endif
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
#ifdef OLD_RESOURCE
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
       int         m_default_frame ; 
#endif
   private:
       const char* m_runresultsdir ;  
   private:
       std::map<std::string, std::string> m_metadata ;  
       std::vector<std::string> m_detector_types ; 
       std::vector<std::string> m_resource_types ; 
};

#include "OKCORE_TAIL.hh"

