#pragma once

#include <string>
#include <vector>

template <typename> class OpticksCfg ;

class TorchStepNPY ; 

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

       static const char* cerenkov_ ;
       static const char* scintillation_ ;
       static const char* torch_ ;
         
       static const char* OTHER_ ;
       static const char* SourceType(int code);
       static unsigned int SourceCode(const char* type);
       static const char* Flag(const unsigned int flag);
       static std::string FlagSequence(const unsigned long long seqhis);
   public:
       Opticks();
       OpticksCfg<Opticks>* getCfg();
       TorchStepNPY* makeSimpleTorchStep();
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
};

inline Opticks::Opticks() 
   :
    m_cfg(NULL)
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

inline void Opticks::init()
{
}

inline void Opticks::configureS(const char* name, std::vector<std::string> values)
{
}

inline void Opticks::configureI(const char* name, std::vector<int> values)
{
}


 
