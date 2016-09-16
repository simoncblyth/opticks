#pragma once

#include <map>
#include <string>
#include <cstring>
#include <vector>

#include "NGLM.hpp"


struct SArgs ; 
template <typename> class NPY ;
template <typename> class OpticksCfg ;

class BDynamicDefine ; 
class TorchStepNPY ; 
class NState ;
class Parameters ; 
class NPropNames ; 
class Timer ; 
class Types ;
class Typ ;
class Index ; 

class Opticks ; 
class OpticksEventSpec ;
class OpticksEvent ;
class OpticksMode ;
class OpticksResource ; 
class OpticksColors ; 
class OpticksQuery; 
class OpticksFlags ;
class OpticksAttrSeq ;
class OpticksProfile ;
class OpticksAna ;

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

class OKCORE_API Opticks {
       friend class OpticksCfg<Opticks> ; 
       friend class OpticksRun ; 
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
       static unsigned int DOMAIN_LENGTH ; 
       static float        DOMAIN_LOW ; 
       static float        DOMAIN_HIGH ; 
       static float        DOMAIN_STEP ; 
       static glm::vec4    getDefaultDomainSpec();

   public:
       Opticks(int argc=0, char** argv=NULL, const char* argforced=NULL );
   private:
       void init();
   public:
       void configure();  // invoked after commandline parsed
       void Summary(const char* msg="Opticks::Summary");
       void dumpArgs(const char* msg="Opticks::dumpArgs");
       bool hasOpt(const char* name);
       bool operator()(const char* name) const ; 
       void cleanup();
       void postpropagate();
       void ana();
   public:
       // profile ops
       template <typename T> void profile(T label);
       void dumpProfile(const char* msg="Opticks::dumpProfile");
       void saveProfile();
   private:
       void checkOptionValidity();
   public:
       // from OpticksResource
       const char* getDetector();
       const char* getDefaultMaterial();
       bool isJuno();
       bool isDayabay();
       bool isPmtInBox();
       bool isOther();
       bool isValid();
       int  getRC();
       void setRC(int rc); 
   public:
       void prepareInstallCache(const char* dir=NULL);
   public:
       const char* getRNGInstallCacheDir();
       const char* getInstallPrefix();
       const char* getMaterialPrefix();
       std::string getObjectPath(const char* name, unsigned int ridx, bool relative=false);
       const char* getDAEPath();
       const char* getGDMLPath();
       const char* getIdPath();
       const char* getIdFold();
       const char* getDetectorBase();
       const char* getMaterialMap();
       const char* getLastArg();
       int         getLastArgInt();
       int         getInteractivityLevel();
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
       OpticksCfg<Opticks>* getCfg();
       std::string          getAnaKey();
       OpticksResource*     getResource(); 
   public:
       OpticksQuery*        getQuery(); 
       OpticksColors*       getColors(); 
       OpticksFlags*        getFlags(); 
       OpticksAttrSeq*      getFlagNames();
       std::map<unsigned int, std::string> getFlagNamesMap();
   public:
       Types*               getTypes();
       Typ*                 getTyp();
   public:
       Timer*               getTimer();
       Parameters*          getParameters();
       NState*              getState();
   public:
       int                  getMultiEvent();
       unsigned int         getSourceCode();
       char                 getEntryCode();    // G:generate S:seedTest T:trivial
       const char*          getEntryName();    
       bool                 isTrivial();
       bool                 isSeedtest();

       bool                 isNoInputGensteps();          // eg when loading a prior propagation
       bool                 isLiveGensteps();             // --live option indicating get gensteps from G4 directly
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
       unsigned int getRngMax();
       unsigned int getBounceMax();
       unsigned int getRecordMax();
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
   public:
       // methods required by BCfg listener classes
       void configureF(const char* name, std::vector<float> values);
       void configureI(const char* name, std::vector<int> values);
       void configureS(const char* name, std::vector<std::string> values);
   private:
       void setCfg(OpticksCfg<Opticks>* cfg);
   private:
       Opticks*             m_ok ;   // for OK_PROFILE 
       SArgs*               m_sargs ; 
       int                  m_argc ; 
       char**               m_argv ; 
       const char*          m_envprefix ;
       const char*          m_materialprefix ;
   private:
       OpticksEventSpec*    m_spec ; 
       OpticksEventSpec*    m_nspec ; 
       OpticksResource*     m_resource ; 
       NState*              m_state ; 
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
       Parameters*          m_parameters ; 
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
       OpticksProfile*      m_profile ; 
       OpticksAna*          m_ana ; 
       int                  m_rc ; 
       unsigned             m_tagoffset ; 
   private:
       glm::uvec4           m_size ; 
       glm::uvec4           m_position ; 


};

#include "OKCORE_TAIL.hh"


