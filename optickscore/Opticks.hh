#pragma once

#include <map>
#include <string>
#include <cstring>
#include <vector>

#include "NGLM.hpp"


class SLog ; 
struct SArgs ; 
template <typename> class NPY ;
template <typename> class OpticksCfg ;

class BDynamicDefine ; 
class TorchStepNPY ; 
class NState ;
class NSensorList ;

struct NSlice ;
struct NSceneConfig ; 
struct NLODConfig ; 
struct NSnapConfig ; 

class NParameters ; 
class NPropNames ; 
class Timer ; 
class Types ;
class Typ ;
class Index ; 

class Opticks ; 
class BOpticksKey ; 
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

#define OK_PROFILE(s) \
    { \
       if(m_ok)\
       {\
          m_ok->profile((s)) ;\
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
   public:
       static const float F_SPEED_OF_LIGHT ;  // mm/ns
   public:
       // TODO: move into OpticksMode
       static const char* COMPUTE_ARG_ ; 

   public:
       static NPropNames* G_MATERIAL_NAMES ;
       static const char* Material(const unsigned int mat);
       static std::string MaterialSequence(const unsigned long long seqmat);
   public:
       // wavelength domain
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
   private:
       static Opticks*     fInstance ;  
   public:
       static Opticks* GetInstance();
       static bool     HasInstance();
   public:
       Opticks(int argc=0, char** argv=NULL, const char* argforced=NULL );
   private:
       void init();
   public:
       void configure();  // invoked after commandline parsed
   private:
       void configureCheckGeometryFiles()  ; // non-const may setExit 
       void configureGeometryHandling() ; 
   public:
       std::string brief();
       void dump(const char* msg="Opticks::dump") ;
       void Summary(const char* msg="Opticks::Summary");
       void dumpArgs(const char* msg="Opticks::dumpArgs");
       void dumpParameters(const char* msg="Opticks::dumpParameters");
       bool hasOpt(const char* name) const ;
       bool operator()(const char* name) const ; 
       void cleanup();
       void postpropagate();
       void ana();

   public:
       // profile ops
       template <typename T> void profile(T label);
       void dumpProfile(const char* msg="Opticks::dumpProfile", const char* startswith=NULL, const char* spacewith=NULL, double tcut=0 );
       void saveProfile();
   private:
       void checkOptionValidity();
   public:
       // from OpticksResource
       const char* getDetector();
       const char* getDefaultMaterial();
       const char* getExampleMaterialNames();
       bool isJuno();
       bool isDayabay();
       bool isPmtInBox();
       bool isOther();
       bool isValid();
       bool hasVolnames() const ; 
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
       void prepareInstallCache(const char* dir=NULL);
   public:
       const char* getRNGInstallCacheDir();
       const char* getInstallPrefix();
       const char* getMaterialPrefix();
       std::string getObjectPath(const char* name, unsigned int ridx, bool relative=false);
       const char* getDAEPath();
       const char* getGDMLPath();

       NSensorList* getSensorList();
       const char* getIdPath();
       const char* getIdFold();
       const char* getDetectorBase();
       const char* getMaterialMap();
       const char* getLastArg();
       int         getLastArgInt();
       int         getInteractivityLevel();
       std::string getArgLine();
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
       void setGeocache(bool geocache=true);
       bool isGeocache();
       void setInstanced(bool instanced=true);
       bool isInstanced();
       void setIntegrated(bool integrated=true);  // used to distinguish OKG4 usage 
       bool isIntegrated();
   public:
       std::string getRelativePath(const char* path); 
   public:
       void setModeOverride(unsigned int mode);
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
       int   getStack() const ;
       int getMeshVerbosity() const ;
       const char* getDbgMesh() const ;
       float getFxRe();
       float getFxAb();
       float getFxSc();
   public:
       // from resource
       const char* getSensorSurface(); 
   public:
       // see GScene, NScene, NGLTF
       const char* getGLTFPath() const ;   // <- standard above geocache position next to the .gdml and .dae
       const char* getGLTFBase() const ;   
       const char* getGLTFName() const ;  
       bool        hasGLTF() const ; 
       const char* getG4CodeGenDir() const ;  // search for g4code
       const char* getCacheMetaPath() const ;
   public:
       // from cfg
       int         getGLTF() const ;
       int         getGLTFTarget() const ;
       bool        isGLTF() const ;

       const char* getGLTFConfig();
       NSceneConfig* getSceneConfig();
   public:
       int         getLayout() const ;
   public:
       const char* getGPUMonPath() const ;   
       bool        isGPUMon() const ;  
   public:
       bool        isTest() const ;
       bool        isTestAuto() const ;
       const char* getTestConfig() const ;

   public:
       bool        isG4Snap() const ; 
       const char* getG4SnapConfigString() const ;
   public:
       const char* getSnapConfigString();
       NSnapConfig* getSnapConfig();
       const char* getLODConfigString();
       NLODConfig* getLODConfig();
       int         getLOD();
       int         getTarget() const ;
       int         getAlignLevel() const;
   public:
       NSlice*  getAnalyticPMTSlice();
       bool     isAnalyticPMTLoad();
       unsigned getAnalyticPMTIndex();
       const char* getAnalyticPMTMedium();
   public:
       OpticksCfg<Opticks>* getCfg() const ;
       const char*          getRenderMode() const ;
       const char*          getLVSDName() const ;
       const char*          getCathode() const ;
   public:
       const char*          getPrintIndexString() const ;
       bool                 getPrintIndex(glm::ivec3& idx) const ;
       int                  getPrintIndex(unsigned dim=0) const ; 
       bool                 isPrintIndexLog() const ; 
       bool                 isG4CodeGen() const ;
       bool                 isPrintEnabled() const ;
       bool                 isXAnalytic() const ;
   public:
       const char*          getDbgIndex() const ;
       const char*          getDbgCSGPath();
       unsigned             getSeed() const ; 
       int                  getLoadVerbosity() const ; 
       int                  getImportVerbosity() const ; 
   public:
       // from cfg
       const char*          getG4GunConfig() const ;
       const char*          getAnaKey() const ;
       const char*          getAnaKeyArgs() const ;
   public:
       OpticksResource*     getResource(); 
       OpticksRun*          getRun(); 
   public:
       OpticksQuery*        getQuery(); 
       OpticksColors*       getColors(); 
       OpticksFlags*        getFlags(); 
       OpticksAttrSeq*      getFlagNames();
       std::map<unsigned int, std::string> getFlagNamesMap();
   public:
       // from OpticksDbg --dindex and --oindex options  
       // NB these are for cfg4 debugging  (Opticks uses different approach with --pindex option)
       NPY<unsigned>* getMaskBuffer() const ; 
       const std::vector<unsigned>&  getMask() const ;
       unsigned getMaskIndex(unsigned idx) const ;  // original pre-masked index OR idx if no mask 
       bool hasMask() const ; 

       bool isDbgPhoton(unsigned record_id);
       bool isOtherPhoton(unsigned record_id);
       bool isMaskPhoton(unsigned record_id);

       bool isDbgPhoton(int event_id, int track_id);
       bool isOtherPhoton(int event_id, int track_id);
       bool isMaskPhoton(int event_id, int track_id);

       const std::vector<unsigned>&  getDbgIndex();
       const std::vector<unsigned>&  getOtherIndex();

       unsigned getNumDbgPhoton() const ;
       unsigned getNumOtherPhoton() const ; 
       unsigned getNumMaskPhoton() const ; 
   public:
       Types*               getTypes();
       Typ*                 getTyp();
   public:
       Timer*               getTimer();
       NParameters*          getParameters();
       NState*              getState();
   public:
       int                  getMultiEvent();
       int                  getRestrictMesh();
       unsigned int         getSourceCode();
       char                 getEntryCode();    // G:generate S:seedTest T:trivial
       const char*          getEntryName();    
       bool                 isTrivial();
       bool                 isSeedtest();

       bool                 isNoInputGensteps();          // eg when loading a prior propagation
       bool                 isLiveGensteps();             // --live option indicating get gensteps from G4 directly
       bool                 isEmbedded() const ;          // --embedded option indicating get gensteps via OpMgr API 


       bool                 isFabricatedGensteps();       // TORCH or MACHINERY source
       const char*          getSourceType();
       const char*          getEventTag();
       const char*          getEventDir();  // tag directory 
       const char*          getEventFold(); // one level above the tag directory 
       int                  getEventITag(); 
       const char*          getEventCat();
       const char*          getUDet();
   public:
       std::string          getPreferenceDir(const char* type, const char* subtype);
   public:
       std::string          getGenstepPath();
       bool                 existsGenstepPath();
       NPY<float>*          loadGenstep();
       TorchStepNPY*        makeSimpleTorchStep();
       OpticksEventSpec*    getEventSpec();
       OpticksEvent*        makeEvent(bool ok=true, unsigned tagoffset=0); 
       OpticksEvent*        loadEvent(bool ok=true, unsigned tagoffset=0); 
       BDynamicDefine*      makeDynamicDefine();
   public:
       // load precooked indices
       Index*               loadHistoryIndex();
       Index*               loadMaterialIndex();
       Index*               loadBoundaryIndex();
   public:
       const glm::vec4&  getTimeDomain();
       const glm::vec4&  getSpaceDomain();
       const glm::vec4&  getWavelengthDomain();
       const glm::ivec4& getSettings();
   public:
       float getTimeMin();
       float getTimeMax();
       float getAnimTimeMax();
   public:
       // screen frame 
       const glm::uvec4& getSize();
       const glm::uvec4& getPosition();
   public:
       void setSpaceDomain(float x, float y, float z, float w);  // triggers configureDomains setting time and wavelength domains too
       void setSpaceDomain(const glm::vec4& pd); 
       std::string description();
   private:
       void defineEventSpec();
       void configureDomains();
   public:
       unsigned getNumPhotonsPerG4Event();
       unsigned getRngMax();
       unsigned getBounceMax();
       unsigned getRecordMax();
       float        getEpsilon();
   public:
       void setExit(bool exit=true);
   public:
       bool hasArg(const char* arg);
       bool isExit();
       bool isRemoteSession();
   public:
       int    getArgc();
       char** getArgv();
       char*  getArgv0();
   public:
       // attempt to follow request,  but constrain to compute when remote session
       bool isCompute();
       bool isInterop();
       bool isCfG4();   // needs manual override to set to CFG4_MODE
       bool isProduction();

       bool isAlign() const ; // --align
       bool isDbgNoJumpZero() const ; // --dbgnojumpzero
       bool isDbgSkipClearZero() const ; // --dbgskipclearzero
       bool isDbgKludgeFlatZero() const ; // --dbgkludgeflatzero

       bool isReflectCheat() const ;
       bool isSave() const ;
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
       bool isDbgClose() const ; 
   public:
       bool isInternal() const ; 
       bool isDumpEnv() const ; 
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
       int                  m_argc ; 
       char**               m_argv ; 
       bool                 m_dumpenv ; 
       bool                 m_production ; 
       OpticksProfile*      m_profile ; 
       const char*          m_materialprefix ;
   private:
       unsigned             m_photons_per_g4event ;
   private:
       OpticksEventSpec*    m_spec ; 
       OpticksEventSpec*    m_nspec ; 
       OpticksResource*     m_resource ; 
       NState*              m_state ; 
       NSlice*              m_apmtslice ; 
       const char*          m_apmtmedium ; 
   private:
       bool             m_exit ; 
       bool             m_compute ; 
       bool             m_geocache ; 
       bool             m_instanced ; 
       bool             m_integrated ; 
       const char*      m_lastarg ; 

   private:
       OpticksCfg<Opticks>* m_cfg ; 
       Timer*               m_timer ; 
       NParameters*          m_parameters ; 
       NSceneConfig*        m_scene_config ; 
       NLODConfig*          m_lod_config ; 
       NSnapConfig*         m_snap_config ; 
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

       OpticksMode*         m_mode ; 
       OpticksRun*          m_run ; 
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

};

#include "OKCORE_TAIL.hh"


