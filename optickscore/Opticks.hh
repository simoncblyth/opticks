#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <glm/glm.hpp>

template <typename> class OpticksCfg ;

class TorchStepNPY ; 
class OpticksEvent ;
class NLog ;
class NState ;
class Parameters ; 
class OpticksResource ; 
class NPropNames ; 

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
       static const char* ZERO_ ;
       static const char* CERENKOV_ ;
       static const char* SCINTILLATION_ ;

       static const char* MISS_ ;
       static const char* BULK_ABSORB_ ;
       static const char* BULK_REEMIT_ ;
       static const char* BULK_SCATTER_ ;
       static const char* SURFACE_DETECT_ ;
       static const char* SURFACE_ABSORB_ ;
       static const char* SURFACE_DREFLECT_ ;
       static const char* SURFACE_SREFLECT_ ;
       static const char* BOUNDARY_REFLECT_ ;
       static const char* BOUNDARY_TRANSMIT_ ;
       static const char* TORCH_ ;

        // unclear how to handle as cannot do parallel wise like TORCH_, G4 only 
       static const char* G4GUN_ ;   

       static const char* NAN_ABORT_ ;
       static const char* BAD_FLAG_ ;
       static const char* OTHER_ ;

       static const char* cerenkov_ ;
       static const char* scintillation_ ;
       static const char* torch_ ;
       static const char* g4gun_ ;
       static const char* other_ ;

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
         
       static const char* SourceType(int code);
       static const char* SourceTypeLowercase(int code);
       static unsigned int SourceCode(const char* type);
       static const char* Flag(const unsigned int flag);
       static std::string FlagSequence(const unsigned long long seqhis);
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
       Opticks(int argc=0, char** argv=NULL, const char* logname="opticks.log", const char* envprefix="OPTICKS_");

   private:
       void init();
       void preargs(int argc, char** argv);
       void preconfigure(int argc, char** argv);
   public:
       void configure();  // invoked after commandline parsed
       void Summary(const char* msg="Opticks::Summary");
       void dumpArgs(const char* msg="Opticks::dumpArgs");
       bool hasOpt(const char* name);
   public:
       // from OpticksResource
       bool isJuno();
       bool isDayabay();
       bool isPmtInBox();
       bool isOther();
       bool isValid();
   public:
       const char* getDAEPath();
       const char* getGDMLPath();
       const char* getIdPath();
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
       Parameters*          getParameters();
       NState*              getState();
       std::string          getModeString();
       const char*          getUDet();
       std::string          getPreferenceDir(const char* type, const char* subtype);
   public:
       TorchStepNPY*        makeSimpleTorchStep();
   public:
       OpticksEvent*        makeEvent(); 
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
       unsigned int getSourceCode();
       std::string getSourceType();
   public:
       bool isCompute();
       bool isInterop();
       bool isCfG4();
   public:
       // methods required by Cfg listener classes
       void configureF(const char* name, std::vector<float> values);
       void configureI(const char* name, std::vector<int> values);
       void configureS(const char* name, std::vector<std::string> values);
   private:
       void configureDomains();
       void setCfg(OpticksCfg<Opticks>* cfg);
   private:
       int                  m_argc ; 
       char**               m_argv ; 

       const char*      m_logname  ; 
       const char*      m_envprefix ;
       OpticksResource* m_resource ; 
       const char*      m_loglevel  ; 
       NLog*            m_log ; 
       NState*          m_state ; 
   private:
       bool             m_compute ; 
       bool             m_geocache ; 
       bool             m_instanced ; 
       const char*      m_lastarg ; 

   private:
       OpticksCfg<Opticks>* m_cfg ; 
       Parameters*          m_parameters ; 
       const char*          m_detector ; 

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



inline Opticks::Opticks(int argc, char** argv, const char* logname, const char* envprefix)
     :
       m_argc(argc),
       m_argv(argv),
       m_logname(strdup(logname)),
       m_envprefix(strdup(envprefix)),
       m_resource(NULL),

       m_loglevel(NULL),
       m_log(NULL),
       m_state(NULL),

       m_compute(false),
       m_geocache(false),
       m_instanced(true),

       m_lastarg(NULL),

       m_cfg(NULL),
       m_parameters(NULL),
       m_detector(NULL),
       m_mode(0u)
{
       init();
}

inline void Opticks::setCfg(OpticksCfg<Opticks>* cfg)
{
    m_cfg = cfg ; 
}
inline OpticksCfg<Opticks>* Opticks::getCfg()
{
    return m_cfg ; 
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





 
