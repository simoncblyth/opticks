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
#include <string>
#include <cstring>
#include <vector>
#include "plog/Severity.h"

#include "NGLM.hpp"

class SLog ; 
class SRngSpec ; 
struct SArgs ; 
class SGeo ; 
class SRenderer ; 

class BDynamicDefine ; 
class BMeta ; 
class BTxt ; 
class BPropNames ; 
class BOpticksKey ; 
class BOpticksResource ; 

template <typename> class NPY ;
class TorchStepNPY ; 
class NState ;

struct NSlice ;
struct NSceneConfig ; 
struct NLODConfig ; 
struct NSnapConfig ; 
class  FlightPath ; 
class  Snap ; 

class Types ;
class Typ ;
class Index ; 
class Composition ; 

template <typename> class OpticksCfg ;
class Opticks ; 
class OpticksEventSpec ;
class OpticksEvent ;
class OpticksRun ;
class OpticksMode ;
class OpticksResource ; 
class OpticksColors ; 
class OpticksQuery; 
class OpticksFlags ;
class OpticksAttrSeq ;
class OpticksProfile ;
class OpticksAna ;
class OpticksDbg ;

class SensorLib ; 

#define OK_PROFILE(s) \
    { \
       if(m_ok)\
       {\
          m_ok->profile((s)) ;\
       }\
    }

#define OKI_PROFILE(s) \
    { \
       if(Opticks::HasInstance())\
       {\
          Opticks::Instance()->profile((s)) ;\
       }\
    }



#include "OpticksPhoton.h"

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
Opticks
========

Handles commandline or envvar user arguments.

**/


class OKCORE_API Opticks {
       friend class OpticksCfg<Opticks> ; 
       friend class OpticksRun ; 
       friend class OpEngine ; 
       friend class CG4 ; 
       friend struct OpticksTest ; 
   public:
       // (GEOCACHE_CODE_VERSION is incremented when code changes invalidate loading old geocache dirs)  
       static const int    GEOCACHE_CODE_VERSION ; 
       static const char*  GEOCACHE_CODE_VERSION_KEY ; 
   public:
       static const plog::Severity LEVEL ;  
       static const float F_SPEED_OF_LIGHT ;  // mm/ns
       static const char* DEFAULT_PFX ; 
   public:
       static bool IsLegacyGeometryEnabled(); 
       static bool IsForeignGeant4Enabled(); 
       static bool IsGeant4EnvironmentDetected(); 

       static const char* OriginGDMLPath() ; 
       static const char* OptiXCachePathDefault(); 
   public:
       static BPropNames* G_MATERIAL_NAMES ;
       static const char* Material(const unsigned int mat);
       static std::string MaterialSequence(const unsigned long long seqmat);
   public:
       // wavelength domain
       static unsigned     DomainLength() ; 
       static const char   DOMAIN_TYPE ; // 'F' or 'C' 
       static unsigned     DOMAIN_LENGTH ; 
       static unsigned     FINE_DOMAIN_LENGTH ; 
       static float        DOMAIN_LOW ; 
       static float        DOMAIN_HIGH ; 
       static float        DOMAIN_STEP ; 
       static float        FINE_DOMAIN_STEP ; 
       static glm::vec4    getDefaultDomainSpec();
       static glm::vec4    getDefaultDomainReciprocalSpec();

       static glm::vec4    getDomainSpec(bool fine=false);
       static glm::vec4    getDomainReciprocalSpec(bool fine=false);
   public:
       static BOpticksKey* GetKey();
       static bool         SetKey(const char* keyspec);
       BOpticksKey*        getKey() const ;  // non-static : the key actually in use, usually the same as GetKey()
       const char*         getKeySpec() const ; 
   private:
       static Opticks*     fInstance ;  
   public:
       static Opticks* Instance();
       static Opticks* Get();  // creates if not existing 
       static bool     HasInstance();
       static bool     HasKey();

   private:
       bool envkey(); 
   public:
       Opticks(int argc=0, char** argv=NULL, const char* argforced=NULL );
   private:
       void init();
       void initResource();
   public:
       void configure();  // invoked after commandline parsed
       bool isConfigured() const ;  
       const char* getCVD() const ;
       const char* getDefaultCVD() const ;
       const char* getUsedCVD() const ;
   private:
       void postconfigure(); 
       void postconfigureCVD() ;
       void postconfigureSize() ;
       void postconfigurePosition() ;
       void postconfigureComposition() ;
       void postconfigureState() ;
       void postconfigureGeometryHandling();
   public:
       std::string brief();
       void dump(const char* msg="Opticks::dump") ;
       void Summary(const char* msg="Opticks::Summary");
       void dumpArgs(const char* msg="Opticks::dumpArgs");
       void dumpMeta(const char* msg="Opticks::dumpMeta") const ;
       void dumpParameters(const char* msg="Opticks::dumpParameters") const ;
       void saveParameters() const ;  // into RunResultsDir
       bool hasOpt(const char* name) const ;
       bool operator()(const char* name) const ; 
       void cleanup();
       void postgeocache();
       void postpropagate();
   public:
       static void Finalize(); 
   public:
       void ana();
       OpticksAna*  getAna() const ; 
   public:
       SensorLib*   getSensorLib() const ;  
       void         initSensorData(unsigned num_sensors); 
   public:
       // profile ops
       void profile(const char* label);
       const glm::vec4& getLastStamp() const  ;

       void dumpProfile(const char* msg="Opticks::dumpProfile", const char* startswith=NULL, const char* spacewith=NULL, double tcut=0 );
       void setProfileDir(const char* dir);
       const char* getProfileDir() const ;   
       void saveProfile();

       unsigned accumulateAdd(const char* label); 
       void     accumulateStart(unsigned idx); 
       void     accumulateStop(unsigned idx); 
       std::string accumulateDesc(unsigned idx);

       void     accumulateSet(unsigned idx, float value); 
       unsigned lisAdd(const char* label); 
       void lisAppend(unsigned idx, double t); 

   private:
       void checkOptionValidity();
   public:
       // from OpticksResource
       bool isValid();

       bool isEnabledLegacyG4DAE() const ;  // --enabled_legacy_g4dae 
       bool isGPartsTransformOffset() const ; // --gparts_transform_offset

       //bool isLocalG4() const ; // --localg4 
   public:
       int  rc() const ;
       void dumpRC() const ;
       int  getRC() const ;
       void setRC(int rc, const char* msg ); 
       const char* getRCMessage() const ; 
   private:
       bool hasCtrlKey(const char* key) const ;

   public:
       // verbosity typically comes from geometry metadata
       unsigned getVerbosity() const ;
       void setVerbosity(unsigned verbosity);
   public:
       const char* getInstallPrefix();

       std::string getObjectPath(const char* name, unsigned int ridx, bool relative=false) const ;
       std::string getObjectPath(const char* name, bool relative=false) const ;
   public:
       const char* getGDMLPath() const ;
       const char* getSrcGDMLPath() const ;
       const char* getOriginGDMLPath() const ;  // formerly getDirectGDMLPath
       const char* getOriginGDMLPathKludged() const ;  
       const char* getCurrentGDMLPath() const ;
   public:
       const char* getDbgGDMLPath() const ; // --dbggdmlpath : used for sneaky GDML exports for debugging 
   public:
       bool        hasGeocache() const ; 
       const char* getIdPath() const ;
       std::string getCSG_GGeoDir() const ;
       const char* getFoundryBase(const char* ekey="CFBASE") const ; 


       const char* getGeocacheDir() const ; // eg ~/.opticks/geocache 
       const char* getGeocacheScriptPath() const ; // eg ~/.opticks/geocache/geocache.sh 
       void writeOutputDirScript(const char* outdir) const ; 

       bool        hasIdPath() const ; 
       const char* getKeyDir() const ;
       const char* getIdFold() const ;


       const char* getLastArg();
       int         getLastArgInt();
       int         getInteractivityLevel() const ;  // from m_mode (OpticksMode)
       std::string getArgLine() const ;
   public:
       unsigned    getOptiXVersion();
       unsigned    getGeant4Version();
   private:
       void setOptiXVersion(unsigned version);
       void setGeant4Version(unsigned version);
   public:
       void setIdPathOverride(const char* idpath_tmp=NULL); // used for saves into non-standard locations whilst testing
       unsigned getTagOffset();
   private:
       void setTagOffset(unsigned tagoffset);   // set by Opticks::makeEvent, used for uniqing profile labels
   public:
       void setGeocacheEnabled(bool geocache=true);
       bool isGeocacheEnabled() const ;
       bool isGeocacheAvailable() const ;

       void setInstanced(bool instanced=true);
       bool isInstanced();
       void setIntegrated(bool integrated=true);  // used to distinguish OKG4 usage 
       bool isIntegrated();
   public:
       std::string formCacheRelativePath(const char* path); 
   public:
       void setDetector(const char* detector); 

   public:
        // from cfg
       const std::string& getSeqMapString() const ;
       void setSeqMapString( const char* seqmap );  // used from OpticksEventAna for --testauto
       bool getSeqMap(unsigned long long& seqhis, unsigned long long& seqval);

       unsigned long long getDbgSeqhis();
       unsigned long long getDbgSeqmat();
       int   getDebugIdx() const ;
       int   getDbgNode() const ;
       int   getDbgMM() const ;
       int   getDbgLV() const ;
       int   getStack() const ;
       int   getMaxCallableProgramDepth() const ;
       int   getMaxTraceDepth() const ;
       int   getUsageReportLevel() const ; 

       int getMeshVerbosity() const ;
       const char* getAccel() const ;
       const char* getDbgMesh() const ;
       float getFxRe();
       float getFxAb();
       float getFxSc();
   public:
       // these two still needed by GInstancer
       const char* getGLTFConfig();
       NSceneConfig* getSceneConfig();
   public:
       const char* getG4CodeGenDir() const ;  // search for g4code
       const char* getCacheMetaPath() const ;
       const char* getGDMLAuxMetaPath() const ;
       const char* getRunCommentPath() const ;

       const char* getGLTFPath() const ;      // output GLTF path
   public:
       // from cfg
       int         getGLTFTarget() const ;

   public:
       int         getLayout() const ;
   public:
       const char* getGPUMonPath() const ;   
       bool        isGPUMon() const ;  
   public:
       // BMeta parameters 
       template <typename T> void set(const char* name, T value);
   public:
       bool        has_arg(const char* arg) const  ; // via PLOG::instance

       void        updateCacheMeta() ; 
       void        appendCacheMeta(const char* key, BMeta* obj);
       void        saveCacheMeta() const ; 
       void        loadOriginCacheMeta() ; 
       BMeta*      getOriginCacheMeta(const char* obj) const ; 
   private:
       const char* getCacheMetaGDMLPath_(const BMeta* origin_cachemeta ) const  ; 
       void        loadOriginCacheMeta_() ; 
   public:
       const BMeta* getGDMLAuxMeta() const  ; 
       void         findGDMLAuxMetaEntries(std::vector<BMeta*>&, const char* key, const char* val ) const ; 
       void        findGDMLAuxValues(std::vector<std::string>& values, const char* k, const char* v, const char* q) const ; // for entries matching (k,v) collect  q values
       unsigned    getGDMLAuxTargetLVNames(std::vector<std::string>& lvnames) const ;
       const char* getGDMLAuxTargetLVName() const ; // returns first name or NULL when none
   public:
       void        dumpCacheMeta(const char* msg="Opticks::dumpCacheMeta") const ; 
       static std::string ExtractCacheMetaGDMLPath(const BMeta* meta) ; 

       const char* getRunComment() const ;
       int         getRunStamp() const ; 
       const char* getRunDate() const ; 
       const char* getRunLabel() const ; 
       static const char* AutoRunLabel(int rtx);
       const char* getRunFolder() const ; 

       const char* getRunResultsDir() const ; // eg /usr/local/opticks/results/OpticksResourceTest/20190422_155146 
       const char* getRuncacheDir() const ;   // eg ~/.opticks/runcache
       const char* getOptiXCacheDirDefault() const ; // eg /var/tmp/simon/OptiXCache where "simon" is username  
   public:
       bool        isTest() const ;
       bool        isTestAuto() const ;
       const char* getTestConfig() const ;
   public:
       bool        isG4Snap() const ; 
       const char* getG4SnapConfigString() const ;
   public:
       const char* getFlightPathDir() const ;
       const char* getFlightConfig() const ;  // --flightconfig  
       const char* getFlightOutDir() const ;  // --flightoutdir
       const char* getSnapOutDir() const ;    // --snapoutdir
       const char* getOutDir() const ;        // --outdir
       void        setOutDir(const char* outdir) ; 

       const char* getOutPrefix(int optix_version_override=0 ) const ; 


       const char* getNamePrefix() const ;    // --nameprefix
       FlightPath* getFlightPath();  // lazy cannot be const  

       std::string getContextAnnotation() const ;
       std::string getFrameAnnotation(unsigned frame, unsigned num_frame, double dt ) const ;
   public:
       const char* getSnapConfigString() const ;
       const char* getSnapOverridePrefix() const ;  // --snapoverrideprefix

       NSnapConfig* getSnapConfig();
       const char*  getSnapPath(int index) ;
       unsigned     getSnapSteps() ;
       void         getSnapEyes(std::vector<glm::vec3>& eyes); 
       Snap*        getSnap(SRenderer* renderer);


       const char* getOutPath(const char* namestem="namestem", const char* ext=".jpg", int index=-1) const ; 
       static int  ExtractIndex(const char* path); 




       const char* getLODConfigString();
       NLODConfig* getLODConfig();
       int         getLOD();
       int         getDomainTarget() const ;   // --domaintarget
       int         getGenstepTarget() const ;  // --gensteptarget
       int         getTarget() const ;         // --target 
       const char* getTargetPVN() const  ;     // --targetpvn

       int         getAlignLevel() const;
   public:
       int         getDefaultFrame() const ; 
   public:
       OpticksCfg<Opticks>* getCfg() const ;
       const char*          getRenderMode() const ;
       const char*          getRenderCmd() const ;  // --rendercmd 
       const char*          getCSGSkipLV() const ;

       const char*          getLVSDName() const ;
       const char*          getCathode() const ;
       const char*          getCerenkovClass() const ;
       const char*          getScintillationClass() const ;
   public:
       const char*          getPrintIndexString() const ;
       bool                 getPrintIndex(glm::ivec3& idx) const ;
       bool                 getAnimTimeRange(glm::vec4& range) const ;
       int                  getPrintIndex(unsigned dim=0) const ; 
       bool                 isPrintIndexLog() const ; 

       unsigned             getWayMask() const ;  // --waymask 3 
       bool                 isWayEnabled() const ;   // --way
       bool                 isSaveGPartsEnabled() const ; // --savegparts
       bool                 isGDMLKludge() const ;   // --gdmlkludge
       bool                 isFineDomain() const ;   // --finedomain
       bool                 isAngularEnabled() const ;  
       void                 setAngularEnabled(bool angular_enabled); 


       bool                 isG4CodeGen() const ;  // --g4codegen
       bool                 isNoSavePPM() const ; // --nosaveppm
       bool                 isNoGPU() const ; // --nogpu
       bool                 isNoG4Propagate() const ;     // --nog4propagate
       bool                 isSaveProfile() const ; // --saveprofile

       bool                 isPrintEnabled() const ;  // --printenabled
       bool                 isExceptionEnabled() const ;  // --exceptionenabled
       bool                 isXAnalytic() const ;      // --xanalytic : --xtriangle option will override an --xanalytic option
       bool                 isXGeometryTriangles() const ;
       bool                 isNoGDMLPath() const ;    // --nogdmlpath
       bool                 isAllowNoKey() const ;    // --allownokey
   public:
       bool                 canDeleteGeoCache() const ; 
       void                 deleteGeoCache() const ; 
       void                 enforceNoGeoCache() const ; 
   public:
       std::string          reportKeyString() const ; 
       void                 reportKey(const char* msg="Opticks::reportKey") const ; 
       std::string          geocacheScriptString(const char* msg) const ; 
       void                 writeGeocacheScript(const char* msg="Opticks::writeGeocacheScript") const ; 
   public:
       const char*          getDbgIndex() const ;
       const char*          getDbgCSGPath();
       unsigned             getSeed() const ; 
       unsigned             getSkipAheadStep() const ;  // --skipaheadstep 1000
       int                  getRTX() const ; 
   public:
       // used by CSGOptiX
       int                     getOneGASIAS() const ;   // --one_gas_ias
       void                    setOneGASIAS(int one_gas_ias) ; 

       int                     getRaygenMode() const ;   // --raygenmode
       void                    setRaygenMode(int raygenmode) ; 

       const char*             getSolidLabel() const ;  // --solid_label    
       std::vector<unsigned>&  getSolidSelection() ; 
       const std::vector<unsigned>& getSolidSelection() const ;
   public:
       int                  getRenderLoopLimit() const ; 
       int                  getAnnoLineHeight() const ;

       int                  getLoadVerbosity() const ; 
       int                  getImportVerbosity() const ; 
   public:
       // from cfg
       const char*          getG4GunConfig() const ;
       const char*          getAnaKey() const ;
       const char*          getAnaKeyArgs() const ;
   public:
       OpticksResource*     getResource() const ; 
       void                 dumpResource() const ; 
       bool                 isKeySource() const ; // name of current executable matches that of the creator of the geocache
       bool                 isKeyLive() const ; 
       OpticksRun*          getRun(); 
   public:
       OpticksQuery*        getQuery(); 
       OpticksColors*       getColors(); 
       OpticksFlags*        getFlags() const ; 
       OpticksAttrSeq*      getFlagNames();
       std::map<unsigned int, std::string> getFlagNamesMap();
   public:
       // from OpticksDbg --dindex and --oindex options  
       // NB these are for cfg4 debugging  (Opticks uses different approach with --pindex option)
       NPY<unsigned>* getMaskBuffer() const ; 
       const std::vector<unsigned>&  getMask() const ;
       unsigned getMaskIndex(unsigned idx) const ;  // original pre-masked index OR idx if no mask 
       bool hasMask() const ; 
       unsigned getMaskSize() const ; 

       unsigned getDbgHitMask() const ;  // --dbghitmask=TO,BT,SD 

       bool isDbgPhoton(unsigned record_id) const ;
       bool isOtherPhoton(unsigned record_id) const ;
       bool isMaskPhoton(unsigned record_id) const ;
       bool isX4PolySkip(unsigned lvIdx) const ;
       bool isX4BalanceSkip(unsigned lvIdx) const ; 
       bool isX4NudgeSkip(unsigned lvIdx) const ; 
       bool isX4TubsNudgeSkip(unsigned lvIdx) const ; 
       bool isX4PointSkip(unsigned lvIdx) const ; 


       bool isCSGSkipLV(unsigned lvIdx) const ;          // --csgskiplv
       unsigned getNumCSGSkipLV() const ;

       bool isDeferredCSGSkipLV(unsigned lvIdx) const ;  // --deferredcsgskiplv
       unsigned getNumDeferredCSGSkipLV() const ;

       bool isSkipSolidIdx(unsigned lvIdx) const ;   // --skipsolidname

       bool isX4SkipSolidName(const char* soname) const;   // --x4skipsolidname  : EARLIER SKIPPING THAT ABOVE 

       unsigned long long getEMM() const ;
       bool isEnabledMergedMesh(unsigned mm) const ;
       const char* getEnabledMergedMesh() const  ; 

       unsigned getInstanceModulo(unsigned mm) const ;


       bool isDbgPhoton(int event_id, int track_id);
       bool isOtherPhoton(int event_id, int track_id);
       bool isMaskPhoton(int event_id, int track_id);

       bool isGenPhoton(int gen_id);

       const std::vector<unsigned>&  getDbgIndex();
       const std::vector<unsigned>&  getOtherIndex();
       const std::vector<unsigned>&  getGenIndex();

       unsigned getNumDbgPhoton() const ;
       unsigned getNumOtherPhoton() const ; 
       unsigned getNumGenPhoton() const ;
       unsigned getNumMaskPhoton() const ; 
   public:
       Types*               getTypes();
       Typ*                 getTyp();
   public:
       OpticksProfile*      getProfile() const ;
       BMeta*               getParameters() const ;
       NState*              getState() const ;
   public:
       int                  getMultiEvent() const ;
       int                  getCameraType() const ;
       int                  getGenerateOverride() const ;
       int                  getPropagateOverride() const ;
   public:
       unsigned int         getSourceCode() const ;
       const char*          getSourceType() const ;
   public:
       bool                 isG4GUNGensteps() const ;       // G4GUN source
       bool                 isFabricatedGensteps() const ;       // TORCH or MACHINERY source
       bool                 isNoInputGensteps() const ;          // eg when loading a prior propagation
       bool                 isLiveGensteps() const ;             // --live option indicating get gensteps from G4 directly
       bool                 isNoPropagate() const ;          //  --nopropagate
   public:
       bool                 isEmbedded() const ;    // --embedded option indicating get gensteps via OpMgr API 
       bool                 hasKey() const ;       // distinguishes direct from legacy mode
       bool                 isDirect() const ; 
       bool                 isLegacy() const ; 
       std::string          getLegacyDesc() const ; 
   public:
       char                 getEntryCode() const ;    // G:generate S:seedTest T:trivial
       const char*          getEntryName() const ;    
       bool                 isTrivial() const ;
       bool                 isSeedtest() const ;
   public:
       const char*          getEventPfx() const ;
       const char*          getEventTag() const ;
       const char*          getEventDir() const ;  // tag directory 
       const char*          getEventFold() const ; // one level above the tag directory 
       int                  getEventITag() const ; 
       const char*          getEventCat() const ;
       const char*          getEventDet() const ;
       const char*          getInputUDet() const ;
   private: 
   public:
       std::string          getPreferenceDir(const char* type, const char* subtype) const ;
       std::string          getFlightInputDir() const ;  // $HOME/.opticks/flight 
       std::string          getFlightInputPath(const char* name="RoundaboutXY") const ;  // eg $HOME/.opticks/flight/RoundaboutXY.npy 
   public:
       NPY<float>*          findGensteps(unsigned tagoffset) const ; 
   public:
       // used from OpticksGen
       bool                 existsLegacyGenstepPath() const ;
       const char*          getLegacyGenstepPath() const ; 
       NPY<float>*          loadLegacyGenstep() const ;
       const char*          getDirectGenstepPath(unsigned tagoffset) const ; 
       const char*          getDebugGenstepPath(unsigned tagoffset) const ; 

       bool isDbgGSImport() const ;  // --dbggsimport
       bool isDbgGSSave() const ;  // --dbggssave
       bool isDbgGSLoad() const ;  // --dbggsload
   public:
       const char*          getPVName() const ; // --pvname
       const char*          getBoundary() const ; // --boundary
       const char*          getMaterial() const ; // --material
       bool                 isLarge() const ; // --large 
       bool                 isMedium() const ; // --medium 
   public:
       const std::vector<std::string>&  getArgList() const ; // --arglist
   private:
       bool                 existsDirectGenstepPath(unsigned tagoffset) const ;
       bool                 existsDebugGenstepPath(unsigned tagoffset) const ;
       // sneaky genstep saving/loading for debugging 

       NPY<float>*          loadDirectGenstep(unsigned tagoffset) const ;
       NPY<float>*          loadDebugGenstep(unsigned tagoffset) const ;

       NPY<float>*          load(const char* path) const ;
   public:
       TorchStepNPY*        makeSimpleTorchStep(unsigned gencode);
   public:
       OpticksEventSpec*    getEventSpec();
       OpticksEvent*        makeEvent(bool ok=true, unsigned tagoffset=0); 
       OpticksEvent*        loadEvent(bool ok=true, unsigned tagoffset=0); 
       BDynamicDefine*      makeDynamicDefine();
   public:
       // via m_run
       void createEvent(NPY<float>* gensteps, char ctrl) ;
       void createEvent(unsigned tagoffset, char ctrl) ;
       void saveEvent(char ctrl) ;
       void resetEvent(char ctrl) ;
       OpticksEvent*        getEvent(char ctrl) const ;   
       OpticksEvent*        getEvent() const ;   
       OpticksEvent*        getG4Event() const ; 
   public:
       // load precooked indices
       Index*               loadHistoryIndex();
       Index*               loadMaterialIndex();
       Index*               loadBoundaryIndex();
   public:
       const glm::vec4&  getTimeDomain() const ;
       const glm::vec4&  getSpaceDomain() const ;
       const glm::vec4&  getWavelengthDomain() const ;
       const glm::ivec4& getSettings() const ;
   public:
       float getTimeMin() const ;   // 
       float getTimeMax() const ;   // --timemax
       float getAnimTimeMax() const ; // --animtimemax
   public:
       // screen frame 
       const glm::uvec4& getSize() const ;
       unsigned          getWidth() const ;
       unsigned          getHeight() const ;
       unsigned          getDepth() const ;
   public:
       const glm::uvec4& getPosition();
   public:
       void setSpaceDomain(float x, float y, float z, float w);  // triggers postgeometry setting time and wavelength domains too
       void setSpaceDomain(const glm::vec4& pd); 
       std::string description() const ;
       std::string desc() const ;
       std::string export_() const ;
   private:
       void setupTimeDomain(float extent); 
       void postgeometry();
       void defineEventSpec();
       void configureDomains();
   public:
       unsigned getNumPhotonsPerG4Event() const ;
       unsigned getManagerMode() const ;
   public:
       const char*        getRNGDir() const ;
       unsigned           getRngMax()   const ; 
       unsigned long long getRngSeed()  const ;
       unsigned long long getRngOffset() const ; 
       const char*        getCURANDStatePath(bool assert_readable=true) const ; 
   public:
       unsigned getBounceMax();
       unsigned getRecordMax();
       float    getEpsilon() const ;
       float    getPixelTimeScale() const ;
       int      getCurFlatSigInt() const ; 
       int      getBoundaryStepSigInt() const ; 
   public:
       void setExit(bool exit=true);
   public:
       bool hasArg(const char* arg) const ;
       bool isExit();
   public:
       int    getArgc() const ;
       char** getArgv() const ;
       char*  getArgv0() const ;
       void   dumpArgv(const char* msg="Opticks::dumpArgv") const ; 
   public:
       bool isRemoteSession() const ;
       // attempt to follow request,  but constrain to compute when remote session
       bool isCompute() const ;
       bool isInterop() const ;

       bool isUTailDebug() const ; // --utaildebug
       bool isProduction() const ; // --production

       bool isAlign() const ; // --align
       bool isDbgNoJumpZero() const ; // --dbgnojumpzero
       bool isDbgFlat() const ; // --dbgflat
       bool isDbgSkipClearZero() const ; // --dbgskipclearzero
       bool isDbgKludgeFlatZero() const ; // --dbgkludgeflatzero
       bool isDbgTex() const ; // --dbgtex
       bool isDbgEmit() const ; // --dbgemit
       bool isDbgDownload() const ; // --dbgdownload
       bool isDbgHit() const ; // --dbghit
       bool isDumpHit() const ; // --dumphit
       bool isDumpHiy() const ; // --dumphiy
       bool isDumpSensor() const ; // --dumpsensor
       bool isSaveSensor() const ; // --savesensor
       bool isDumpProfile() const ; // --dumpprofile
       bool isDbgGeoTest() const ; // --dbggeotest

       bool isReflectCheat() const ;
    public:
        bool getSaveDefault() const ;   // --save is trumped by --nosave 
        void postconfigureSave(); 
        void setSave(bool save);        // code level override of the above default from commandline config 
        bool isSave() const ;
    public:
       bool isLoad() const;
       bool isTracer() const;
       bool isRayLOD() const ; // raytrace LOD via OptiX selector based on ray origin wrt instance position 
       bool isMaterialDbg() const ; 
       bool isDbgAnalytic() const ; 
       bool isDbgSurf() const ; 
       bool isDbgBnd() const ; 
       bool isDbgRec() const ; 
       bool isDbgZero() const ; 
       bool isRecPoi() const ; 
       bool isRecPoiAlign() const ; 
       bool isRecCf() const ; 
       bool isDbgTorch() const ; 
       bool isDbgSource() const ; 
       bool isDbgAim() const ;   // --dbgaim
       bool isDbgClose() const ; 
   public:
       bool isInternal() const ; 
       bool isDumpEnv() const ; 
   public:
       // set by OGLRap.Frame 
       static void SetFrameRenderer(const char* renderer); 
       void setFrameRenderer(const char* renderer); 
       const char*  getFrameRenderer() const ; 
       Composition* getComposition() const ;  
   public:
       void        setGeo( const SGeo* geo ); 
       const SGeo* getGeo() const ; 
   private:
       void setInternal(bool internal=true); 
   public:
       // methods required by BCfg listener classes
       void configureF(const char* name, std::vector<float> values);
       void configureI(const char* name, std::vector<int> values);
       void configureS(const char* name, std::vector<std::string> values);
   private:
       void setCfg(OpticksCfg<Opticks>* cfg);
   private:
       SLog*                m_log ;
       Opticks*             m_ok ;   // for OK_PROFILE 
       SArgs*               m_sargs ; 
       const SGeo*          m_geo ; 
       int                  m_argc ; 
       char**               m_argv ; 
       const char*          m_lastarg ; 
       OpticksMode*         m_mode ; 
       Composition*         m_composition ; 

       bool                 m_dumpenv ; 
       bool                 m_allownokey ; 
       bool                 m_envkey ; 
       bool                 m_production ; 
       OpticksProfile*      m_profile ; 
       bool                 m_profile_enabled ; 

   private:
       unsigned             m_photons_per_g4event ;
   private:
       OpticksEventSpec*    m_spec ; 
       OpticksEventSpec*    m_nspec ; 
       OpticksResource*     m_resource ; 
       BOpticksResource*    m_rsc ; 
       bool                 m_nogdmlpath ; // --nogdmlpath
       const char*          m_origin_gdmlpath ; // formerly m_direct_gdmlpath
       const char*          m_origin_gdmlpath_kludged ; 

       int                  m_origin_geocache_code_version ; 
       NState*              m_state ; 
       NSlice*              m_apmtslice ; 
       const char*          m_apmtmedium ; 
   private:
       bool             m_exit ; 
       bool             m_compute ; 
       bool             m_geocache ; 
       bool             m_instanced ; 
       bool             m_integrated ; 

   private:
       bool                 m_configured ; 
       bool                 m_angular_enabled ; 
       OpticksCfg<Opticks>* m_cfg ; 

       BMeta*               m_parameters ; 
       BTxt*                m_runtxt ;  
       BMeta*               m_cachemeta ;  
       BMeta*               m_origin_cachemeta ;  
       NSceneConfig*        m_scene_config ; 
       NLODConfig*          m_lod_config ; 
       NSnapConfig*         m_snapconfig ; 

       FlightPath*          m_flightpath ;  
       Snap*                m_snap ; 
   private:
       const char*          m_detector ; 
       unsigned             m_event_count ; 
   private:
       bool                 m_domains_configured ;  
       glm::vec4            m_time_domain ; 
       glm::vec4            m_space_domain ; 
       glm::vec4            m_wavelength_domain ; 

   private:
       glm::ivec4       m_settings ; 
       //NB avoid duplication between here and OpticksCfg , only things that need more control need be here

       OpticksRun*          m_run ;   // actually used for dual running 
       OpticksAna*          m_ana ; 
       OpticksDbg*          m_dbg ; 

       int                  m_rc ; 
       const char*          m_rcmsg ; 
       unsigned             m_tagoffset ; 
   private:
       glm::uvec4           m_size ; 
       glm::uvec4           m_position ; 
       unsigned             m_verbosity ; 
       bool                 m_internal ; 
       const char*          m_frame_renderer ; 
       SRngSpec*            m_rngspec ; 
       SensorLib*           m_sensorlib ; 
       int                  m_one_gas_ias ; 
       int                  m_raygenmode ; 
       std::vector<unsigned>  m_solid_selection ; 
       bool                 m_save ; 
       const char*          m_outdir ; 

};

#include "OKCORE_TAIL.hh"


