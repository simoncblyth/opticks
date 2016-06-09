#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <glm/glm.hpp>

template <typename> class OpticksCfg ;

class TorchStepNPY ; 
class BLog ;
class NState ;
class Parameters ; 
class NPropNames ; 
class Timer ; 

class Types ;
class Typ ;

class Index ; 
class OpticksEvent ;
class OpticksResource ; 
class OpticksColors ; 
class OpticksQuery; 
class OpticksFlags ;

class OpticksAttrSeq ;

 

#include "OpticksPhoton.h"

class Opticks {
       friend class OpticksCfg<Opticks> ; 
   public:
       static const char* COMPUTE ; 
   public:
       enum {
               GEOCODE_ANALYTIC = 'A',   
               GEOCODE_TRIANGULATED = 'T',  
               GEOCODE_SKIP = 'K'
            } ;
   public:
       enum { 
              e_shift   = 1 << 0,  
              e_control = 1 << 1,  
              e_option  = 1 << 2,  
              e_command = 1 << 3 
            } ; 
       static std::string describeModifiers(unsigned int modifiers);
       static bool isShift(unsigned int modifiers);
       static bool isControl(unsigned int modifiers);
       static bool isCommand(unsigned int modifiers);
       static bool isOption(unsigned int modifiers);

   public:
       static const float F_SPEED_OF_LIGHT ;  // mm/ns

       static const char* BNDIDX_NAME_ ;
       static const char* SEQHIS_NAME_ ;
       static const char* SEQMAT_NAME_ ;

       static const char* COMPUTE_MODE_ ;
       static const char* INTEROP_MODE_ ;
       static const char* CFG4_MODE_ ;

       enum {
                COMPUTE_MODE = 0x1 << 1, 
                INTEROP_MODE = 0x1 << 2, 
                CFG4_MODE = 0x1 << 3
            }; 
         
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
       Opticks(int argc=0, char** argv=NULL, const char* envprefix="OPTICKS_");
   private:
       void init();
   public:
       void configure();  // invoked after commandline parsed
       void Summary(const char* msg="Opticks::Summary");
       void dumpArgs(const char* msg="Opticks::dumpArgs");
       bool hasOpt(const char* name);
       void cleanup();
   public:
       // from OpticksResource
       const char* getDetector();
       bool isJuno();
       bool isDayabay();
       bool isPmtInBox();
       bool isOther();
       bool isValid();
   public:
       const char* getInstallPrefix();
       std::string getObjectPath(const char* name, unsigned int ridx, bool relative=false);
       const char* getDAEPath();
       const char* getGDMLPath();
       const char* getIdPath();
       const char* getIdFold();
       const char* getLastArg();
       int         getLastArgInt();
   public:
       void setGeocache(bool geocache=true);
       bool isGeocache();
       void setInstanced(bool instanced=true);
       bool isInstanced();
   public:
       std::string getRelativePath(const char* path); 
   public:
       void setMode(unsigned int mode);
       void setDetector(const char* detector); 

   public:
       OpticksCfg<Opticks>* getCfg();
       OpticksResource*     getResource(); 
   public:
       OpticksQuery*        getQuery(); 
       OpticksColors*       getColors(); 
       OpticksFlags*        getFlags(); 
       OpticksAttrSeq*      getFlagNames();
   public:
       Types*               getTypes();
       Typ*                 getTyp();
   public:
       Timer*               getTimer();
       Parameters*          getParameters();
       NState*              getState();
       std::string          getModeString();
   public:
       unsigned int         getSourceCode();
       const char*          getSourceType();
       const char*          getEventTag();
       const char*          getEventCat();
       const char*          getUDet();
   public:
       std::string          getPreferenceDir(const char* type, const char* subtype);
   public:
       TorchStepNPY*        makeSimpleTorchStep();
   public:
       OpticksEvent*        makeEvent(); 
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
       void setSpaceDomain(const glm::vec4& pd);
       std::string description();
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
       bool isCompute();
       bool isInterop();
       bool isCfG4();
   public:
       // methods required by BCfg listener classes
       void configureF(const char* name, std::vector<float> values);
       void configureI(const char* name, std::vector<int> values);
       void configureS(const char* name, std::vector<std::string> values);
   private:
       void configureDomains();
       void setCfg(OpticksCfg<Opticks>* cfg);
   private:
       int                  m_argc ; 
       char**               m_argv ; 
       const char*          m_envprefix ;
   private:
       OpticksResource*     m_resource ; 
       BLog*                m_log ; 
       NState*              m_state ; 
   private:
       bool             m_exit ; 
       bool             m_compute ; 
       bool             m_geocache ; 
       bool             m_instanced ; 
       const char*      m_lastarg ; 

   private:
       OpticksCfg<Opticks>* m_cfg ; 
       Timer*               m_timer ; 
       Parameters*          m_parameters ; 
   private:
       const char*          m_detector ; 
       const char*          m_tag ; 
       const char*          m_cat ; 
   private:
       glm::vec4            m_time_domain ; 
       glm::vec4            m_space_domain ; 
       glm::vec4            m_wavelength_domain ; 

   private:
       glm::ivec4       m_settings ; 
       //NB avoid duplication between here and OpticksCfg , only things that need more control need be here

       unsigned int         m_mode ; 
   private:
       glm::uvec4           m_size ; 
       glm::uvec4           m_position ; 

};



inline Opticks::Opticks(int argc, char** argv, const char* envprefix)
     :
       m_argc(argc),
       m_argv(argv),
       m_envprefix(strdup(envprefix)),

       m_resource(NULL),
       m_log(NULL),
       m_state(NULL),

       m_exit(false),
       m_compute(false),
       m_geocache(false),
       m_instanced(true),

       m_lastarg(NULL),

       m_cfg(NULL),
       m_timer(NULL),
       m_parameters(NULL),
       m_detector(NULL),
       m_tag(NULL),
       m_cat(NULL),
       m_mode(0u)
{
       init();
}



inline bool Opticks::hasArg(const char* arg)
{
    bool has = false ; 
    for(int i=1 ; i < m_argc ; i++ ) if(strcmp(m_argv[i], arg) == 0) has = true ; 
    return has ; 
}

inline void Opticks::setCfg(OpticksCfg<Opticks>* cfg)
{
    m_cfg = cfg ; 
}
inline OpticksCfg<Opticks>* Opticks::getCfg()
{
    return m_cfg ; 
}

inline Timer* Opticks::getTimer()
{
    return m_timer ; 
}
inline Parameters* Opticks::getParameters()
{
    return m_parameters ; 
}

inline OpticksResource* Opticks::getResource()
{
    return m_resource  ; 
}
inline NState* Opticks::getState()
{
    return m_state  ; 
}

inline const char* Opticks::getLastArg()
{
   return m_lastarg ; 
}


inline void Opticks::setMode(unsigned int mode)
{
    m_mode = mode ; 
}
inline bool Opticks::isCompute()
{
    return (m_mode & COMPUTE_MODE) != 0  ; 
}
inline bool Opticks::isInterop()
{
    return (m_mode & INTEROP_MODE) != 0  ; 
}
inline bool Opticks::isCfG4()
{
    return (m_mode & CFG4_MODE) != 0  ; 
}



inline void Opticks::setGeocache(bool geocache)
{
    m_geocache = geocache ; 
}
inline bool Opticks::isGeocache()
{
    return m_geocache ;
}

inline void Opticks::setInstanced(bool instanced)
{
   m_instanced = instanced ;
}
inline bool Opticks::isInstanced()
{
   return m_instanced ; 
}


inline const glm::vec4& Opticks::getTimeDomain()
{
    return m_time_domain ; 
}
inline const glm::vec4& Opticks::getSpaceDomain()
{
    return m_space_domain ; 
}
inline const glm::vec4& Opticks::getWavelengthDomain()
{
    return m_wavelength_domain ; 
}
inline const glm::ivec4& Opticks::getSettings()
{
    return m_settings ; 
}


inline const glm::uvec4& Opticks::getSize()
{
    return m_size ; 
}
inline const glm::uvec4& Opticks::getPosition()
{
    return m_position ; 
}




inline void Opticks::setDetector(const char* detector)
{
    m_detector = detector ? strdup(detector) : NULL ; 
}
inline void Opticks::setSpaceDomain(const glm::vec4& sd)
{
    m_space_domain.x = sd.x  ; 
    m_space_domain.y = sd.y  ; 
    m_space_domain.z = sd.z  ; 
    m_space_domain.w = sd.w  ; 
}


inline void Opticks::configureS(const char* name, std::vector<std::string> values)
{
}

inline void Opticks::configureI(const char* name, std::vector<int> values)
{
}

inline bool Opticks::isExit()
{
    return m_exit ; 
}
inline void Opticks::setExit(bool exit)
{
    m_exit = exit  ;   
}
 
