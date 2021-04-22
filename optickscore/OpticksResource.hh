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

       void initRunResultsDir();
       void setValid(bool valid);
    public:
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
       std::string getMergedMeshPath(unsigned int ridx);
    public:
       std::string getPreferenceDir(const char* type, const char* udet=NULL, const char* subtype=NULL) const  ; 
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
       bool              m_allownokey ; 
       OpticksQuery*     m_query ;
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
       const char* m_runresultsdir ;  
   private:
       std::map<std::string, std::string> m_metadata ;  
       std::vector<std::string> m_detector_types ; 
       std::vector<std::string> m_resource_types ; 
};

#include "OKCORE_TAIL.hh"

