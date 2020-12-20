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

#include <vector>
#include <map>
#include <string>

#include "plog/Severity.h"
#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class SLog ; 

class BOpticksKey ; 
class BPath ; 
class BMeta ; 
class BResource ; 

/**
BOpticksResource : base class to okc.OpticksResource 
======================================================

TODO: rearrange this, avoid confusing split between okc.OpticksResource and brap.BOpticksResource


TODO: Remove alternative setup paths, leaving just the setupViaKey done in ctor 



Constituents:

BPath m_id
    idpath parser, giving access to the elements including a layout integer 

BOpticksKey m_key 
    used for G4 live running 


OPTICKS_RESOURCE_LAYOUT envvar : for expert use only
-------------------------------------------------------

The envvar changes the layout integer, current default is 1 (the old layout is 0). 
Use higher integers to test new geocache writing or new layouts.  

Setting the envvar to eg 100 enables testing geocache writing 
into an empty directory without disturbing the existing geocache in slot 1.
This works by changing the m_layout integer which changes the paths 
to all the geocache Opticks resources.

::

    OPTICKS_RESOURCE_LAYOUT=100 OKTest -G --gltf 1



THIS NEEDS OVERHAUL TO MINIMISE REPETITION BETWEEN THE 
BRANCHES OF OPERATION : USING RELATIVE APPROACH 


**/

class BRAP_API  BOpticksResource {
    public:
        static BOpticksResource* Get(const char* spec) ; 
        static const char* GetCachePath(const char* rela=NULL, const char* relb=NULL, const char* relc=NULL ); 
    private:
        static const plog::Severity  LEVEL ; 
        static BOpticksResource* fInstance ; 
        static BOpticksResource* Instance() ; 
        static BOpticksResource* Create(const char* spec) ;   // normally NULL indicating use OPTICKS_KEY envvar
    public:
        static const char* LEGACY_GEOMETRY_ENABLED_KEY ; 
        static const char* FOREIGN_GEANT4_ENABLED_KEY ; 
        static const char* DEFAULT_GENSTEP_TARGET_KEY ; 
        static const char* DEFAULT_DOMAIN_TARGET_KEY ; 
        static const char* DEFAULT_TARGET_KEY ; 
        static const char* DEFAULT_KEY_KEY ; 

        static bool IsLegacyGeometryEnabled() ; 
        static bool IsForeignGeant4Enabled() ;   // foreign Geant4 means not managed as an Opticks external

        static bool IsGeant4EnvironmentDetected();    // based on the number of envvar keys of form G4...DATA 
        static int  DefaultGenstepTarget(int fallback); 
        static int  DefaultDomainTarget(int fallback); 
        static int  DefaultTarget(int fallback); 
        static std::string  DefaultKey(); 

    protected:
        static const char* RESULTS_PREFIX_KEY  ; 
        static const char* RESULTS_PREFIX_DEFAULT  ; 
    protected:
        static const char* EVENT_BASE_KEY  ; 
        static const char* EVENT_BASE_DEFAULT  ; 
    protected:
        static const char* INSTALL_PREFIX_KEY  ; 
        static const char* INSTALL_PREFIX_KEY2  ; 
    protected:
        static const char* GEOCACHE_PREFIX_KEY  ; 
        static const char* RNGCACHE_PREFIX_KEY  ; 
        static const char* USERCACHE_PREFIX_KEY  ; 
    protected:
        static const char* G4ENV_RELPATH ; 
        static const char* OKDATA_RELPATH ;
    public:
        static const char* PREFERENCE_BASE  ;
        static const char* EMPTY ; 
        static const char* G4LIVE ; 

    public:
        static const char* InstallPathOKDATA() ;
        static const char* InstallPathG4ENV() ;
        static const char* InstallPath(const char* relpath) ;
   public:
        static const char* MakeSrcPath(const char* srcpath, const char* ext) ;
        static const char* MakeSrcDir(const char* srcpath, const char* sub) ;
        static const char* MakeTmpUserDir_(const char* sub, const char* rel) ;
        static const char* MakeTmpUserDir(const char* sub, const char* rel) ;
        static const char* OptiXCachePathDefault() ; 
        static const char* MakeUserDir(const char* sub, const char* rel) ;

        const char* makeIdPathPath(const char* rela, const char* relb=NULL, const char* relc=NULL, const char* reld=NULL) ;

   private:
        BOpticksResource();
   public:
        virtual ~BOpticksResource();
        void Summary(const char* msg="BOpticksResource::Summary") const ;
        void Brief(const char* msg="BOpticksResource::Brief") const ;
        std::string desc() const ;
    
   public:
        static std::string BuildDir(const char* proj);
        static std::string BuildProduct(const char* proj, const char* name);
        static const char* ResolveResultsPrefix();
        static const char* ResolveEventBase();
        static const char* ResolveInstallPrefix();
        static const char* ResolveGeoCachePrefix();
        static const char* ResolveRngCachePrefix();
        static const char* ResolveUserCachePrefix();
        static const char* OpticksDataDir();
        static const char* OpticksAuxDir();
        static const char* GeocacheDir();
        static const char* RNGCacheDir();
        static const char* RNGDir();
        static const char* RuncacheDir();
        static const char* ShaderDir();
        static const char* ResultsDir();

        static const char* ResourceDir();
        static const char* GenstepsDir();
        static const char* ExportDir();

        static const char* InstallCacheDir();
   private:

        static const char* MakePath( const char* prefix, const char* main, const char* sub );
   public:       
        const char* getInstallPrefix();
        const char* getGeoCachePrefix();
        const char* getRngCachePrefix();
        const char* getUserCachePrefix();
        const char* getInstallDir();

        const char* getOpticksDataDir();
        const char* getGeocacheDir();
        const char* getRuncacheDir();
        const char* getResultsDir();
        const char* getInstallCacheDir();
        const char* getResourceDir();
        const char* getGenstepsDir();
        const char* getExportDir();

        const char* getRNGDir();

        const char* getOptiXCacheDirDefault() const ;
        const char* getTmpUserDir() const ;

        const char* getDebuggingTreedir(int argc, char** argv);
   public:       
        const char* getDebuggingIDPATH();
        const char* getDebuggingIDFOLD();
   public:      
        std::string getIdPathPath(const char* rela, const char* relb=NULL, const char* relc=NULL, const char* reld=NULL ) const ; 
        std::string getGeocachePath(const char* rela, const char* relb=NULL, const char* relc=NULL, const char* reld=NULL) const ;
        std::string getResultsPath(const char* rela, const char* relb=NULL, const char* relc=NULL, const char* reld=NULL) const ;
        std::string getPropertyLibDir(const char* name) const ;

        std::string getObjectPath(const char* name, unsigned int index) const ;
        std::string getObjectPath(const char* name) const ;
        std::string getRelativePath(const char* name, unsigned int index) const ;
        std::string getRelativePath(const char* name) const ;

        std::string getInstallPath(const char* relpath) const ;
        const char* getIdPath() const ;
        const char* getIdFold() const ;  // parent directory of idpath containing g4_00.dae
        void setIdPathOverride(const char* idpath_tmp=NULL);  // used for test saves into non-standard locations

    public:
        const char* getSrcPath() const ;
        const char* getSrcDigest() const ;
        const char* getDAEPath() const ;
        const char* getGDMLPath() const ;
        const char* getSrcGDMLPath() const ;
        const char* getSrcGLTFPath() const ;
        const char* getSrcGLTFBase() const ;
        const char* getSrcGLTFName() const ;
    public:
        const char* getG4CodeGenDir() const ;
        const char* getGDMLAuxMetaPath() const ;
        const char* getCacheMetaPath() const ;
        const char* getRunCommentPath() const ;
        const char* getPrimariesPath() const ;
        const char* getGLTFPath() const ;     // output path 
    public:
        static const char* opticks_geospecific_options ; 
        const BMeta* getGDMLAuxMeta() const  ;
        std::string  getGDMLAuxUserinfo(const char* k) const ;
        const char*  getGDMLAuxUserinfoGeospecificOptions() const ;
    public:
        void         findGDMLAuxMetaEntries(std::vector<BMeta*>&, const char* key, const char* val ) const ;
        void         findGDMLAuxValues(std::vector<std::string>& values, const char* k, const char* v, const char* q) const ; // for entries matching (k,v) collect  q values
        unsigned     getGDMLAuxTargetLVNames(std::vector<std::string>& lvnames) const ;
        const char*  getGDMLAuxTargetLVName() const ; // returns first name or NULL when none
    private:
        BMeta*       LoadGDMLAuxMeta() const ;
    public:
        const char* getIdMapPath() const ;
    public:
        const char* getEventBase() const ; 
        const char* getEventPfx() const ; 
        void setEventBase(const char* rela, const char* relb=NULL) ; 
        void setEventPfx(const char* pfx) ; 
    public:
        BOpticksKey* getKey() const ; 
        const char*  getKeySpec() const ;
        std::string  export_() const ; 
        bool         hasKey() const ;   // distinguishes direct from legacy 
        bool         isKeySource() const ; 
        bool         isKeyLive() const ; // true: when creating geocache from live Geant4 geometry 
    public:
       // used to communicate test geometry config from geometry loading to test event writing 
       // see GGeoTest::initCreateCSG 
       void        setTestCSGPath(const char* testcsgpath);
       const char* getTestCSGPath() const ; 
       void        setTestConfig(const char* testconfig);
       const char* getTestConfig() const ; 
  private:
        void init();
        void initInstallPrefix();
        void initGeoCachePrefix();
        void initRngCachePrefix();
        void initUserCachePrefix();
        void initTopDownDirs();
        void initDebuggingIDPATH(); 
        void initViaKey();
        void initMetadata();

   protected:
        friend struct NSensorListTest ; 
        friend struct NSceneTest ; 
        friend struct HitsNPYTest ; 
        friend struct BOpticksResourceTest ; 
        // only use one setup route
        
   public: 
        bool idNameContains(const char* s) const ;
        std::string formCacheRelativePath(const char* path) const ;

        // unfortunately having 2 routes difficult to avoid, as IDPATH is 
        // more convenient in that a single path yields everything, whereas
        // the OpticksResource geokey stuff needs to go via Src
   private:
        void setSrcDigest(const char* srcdigest);

   protected:
        bool         m_testgeo ;   
        SLog*        m_log ; 
        bool         m_setup ; 
        BOpticksKey* m_key ; 
        BPath*       m_id ; 
        BResource*   m_res ;   
 
   protected:
        int         m_layout ; 
        const char* m_install_prefix ;   // from BOpticksResourceCMakeConfig header

        const char* m_geocache_prefix ;  
        const char* m_rngcache_prefix ;  
        const char* m_usercache_prefix ;  

        const char* m_opticksdata_dir ; 
        const char* m_opticksaux_dir ; 
        const char* m_geocache_dir ; 
        const char* m_rngcache_dir ; 
        const char* m_runcache_dir ; 
        const char* m_results_dir ; 
        const char* m_resource_dir ; 
        const char* m_gensteps_dir ; 
        const char* m_export_dir ; 
        const char* m_installcache_dir ; 
        const char* m_optixcachedefault_dir ; 
        const char* m_rng_dir ; 
        const char* m_tmpuser_dir ; 
   protected:
        const char* m_srcpath ; 
        const char* m_srcfold ; 
        const char* m_srcbase ; 
        const char* m_srcdigest ; 
        const char* m_idfold ; 
        const char* m_idfile ; 
        const char* m_idgdml ; 
        const char* m_idsubd ; 
        const char* m_idname ; 
        const char* m_idpath ; 
        const char* m_idpath_tmp ; 
        const char* m_evtbase ; 
        const char* m_evtpfx ; 
   protected:
        const char* m_debugging_idpath ; 
        const char* m_debugging_idfold ; 
   protected:
        const char* m_daepath ;
        const char* m_gdmlpath ;
        const char* m_srcgdmlpath ;
        const char* m_srcgltfpath ;

        const char* m_idmappath ;
        const char* m_g4codegendir ;
        const char* m_gdmlauxmetapath ; 
        const char* m_cachemetapath ; 
        const char* m_runcommentpath ; 
        const char* m_primariespath ; 
        const char* m_directgensteppath ; 
        const char* m_directphotonspath ; 
        const char* m_gltfpath ;
   protected:
        const char* m_testcsgpath ;
        const char* m_testconfig ;

   private:
        const BMeta* m_gdmlauxmeta ; 
        const BMeta* m_gdmlauxmeta_lvmeta ; 
        const BMeta* m_gdmlauxmeta_usermeta ; 
        const char*  m_opticks_geospecific_options ; 

};

#include "BRAP_TAIL.hh"



