#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <glm/glm.hpp>

template <typename> class OpticksCfg ;

class TorchStepNPY ; 
class NumpyEvt ;

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
       Opticks();
       OpticksCfg<Opticks>* getCfg();
       void setDetector(const char* detector); 

       TorchStepNPY* makeSimpleTorchStep();
       NumpyEvt* makeEvt(); 
       const glm::vec4& getTimeDomain();
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
       const char*          m_detector ; 
       glm::vec4            m_time_domain ; 
};

inline Opticks::Opticks() 
   :
    m_cfg(NULL),
    m_detector(NULL)
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

inline const glm::vec4& Opticks::getTimeDomain()
{
    return m_time_domain ; 
}

inline void Opticks::setDetector(const char* detector)
{
    m_detector = detector ? strdup(detector) : NULL ; 
}





inline void Opticks::configureS(const char* name, std::vector<std::string> values)
{
}

inline void Opticks::configureI(const char* name, std::vector<int> values)
{
}


 
