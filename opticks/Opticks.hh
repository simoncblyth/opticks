#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <glm/glm.hpp>

template <typename> class OpticksCfg ;

class TorchStepNPY ; 
class NumpyEvt ;
class Parameters ; 

#include "OpticksPhoton.h"

class Opticks {
       friend class OpticksCfg<Opticks> ; 
   public:
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
       static const char* NAN_ABORT_ ;
       static const char* BAD_FLAG_ ;
       static const char* OTHER_ ;

       static const char* cerenkov_ ;
       static const char* scintillation_ ;
       static const char* torch_ ;
       static const char* other_ ;
         
       static const char* SourceType(int code);
       static const char* SourceTypeLowercase(int code);
       static unsigned int SourceCode(const char* type);
       static const char* Flag(const unsigned int flag);
       static std::string FlagSequence(const unsigned long long seqhis);
   public:
       // wavelength domain
       static unsigned int DOMAIN_LENGTH ; 
       static float        DOMAIN_LOW ; 
       static float        DOMAIN_HIGH ; 
       static float        DOMAIN_STEP ; 
       static glm::vec4    getDefaultDomainSpec();
   public:
       Opticks();
       OpticksCfg<Opticks>* getCfg();
       Parameters* getParameters();
       void setDetector(const char* detector); 

       TorchStepNPY* makeSimpleTorchStep();
       NumpyEvt* makeEvt(); 
   public:
       const glm::vec4&  getTimeDomain();
       const glm::vec4&  getSpaceDomain();
       const glm::vec4&  getWavelengthDomain();
       const glm::ivec4& getSettings();
   public:
       void setSpaceDomain(const glm::vec4& pd);
       void collectParameters();
   public:
       void setRngMax(unsigned int rng_max);
       void setBounceMax(unsigned int bounce_max);
       void setRecordMax(unsigned int record_max);
   private:
       void updateSettings();
   public:
       unsigned int getRngMax();
       unsigned int getBounceMax();
       unsigned int getRecordMax();
   public:
       unsigned int getSourceCode();
       std::string getSourceType();
   public:
       // methods required by Cfg listener classes
       void configureF(const char* name, std::vector<float> values);
       void configureI(const char* name, std::vector<int> values);
       void configureS(const char* name, std::vector<std::string> values);
   private:
       void init();
       void setCfg(OpticksCfg<Opticks>* cfg);
   private:
       OpticksCfg<Opticks>* m_cfg ; 
       Parameters*          m_parameters ; 
       const char*          m_detector ; 
       glm::vec4            m_time_domain ; 
       glm::vec4            m_space_domain ; 
       glm::vec4            m_wavelength_domain ; 

   private:
       glm::ivec4       m_settings ; 
       unsigned int     m_rng_max ; 
       unsigned int     m_bounce_max ; 
       unsigned int     m_record_max ; 
};

inline Opticks::Opticks() 
   :
    m_cfg(NULL),
    m_parameters(NULL),
    m_detector(NULL),
    m_rng_max(0),
    m_bounce_max(9),
    m_record_max(10)
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



inline void Opticks::updateSettings()
{
    m_settings.x = m_bounce_max ;   
    m_settings.y = m_rng_max ;   
    m_settings.z = 0 ;   
    m_settings.w = m_record_max ;   
}



inline void Opticks::setRngMax(unsigned int rng_max)
{
// default of 0 disables Rng 
// otherwise maximum number of RNG streams, 
// should be a little more than the max number of photons to generate/propagate eg 3e6
    m_rng_max = rng_max ;
    updateSettings();
}
inline unsigned int Opticks::getRngMax()
{
    return m_rng_max ;
}

inline void Opticks::setBounceMax(unsigned int bounce_max)
{
    m_bounce_max = bounce_max ;
    updateSettings();
}
inline unsigned int Opticks::getBounceMax()
{
    return m_bounce_max ;
}

inline void Opticks::setRecordMax(unsigned int record_max)
{
    m_record_max = record_max ;
    updateSettings();
}
inline unsigned int Opticks::getRecordMax()
{
    return m_record_max ;
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
 
