#pragma once
#include <string>
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksMode
===============

Constructor resident of Opticks instanciated very early prior to configuration.

**/

class Opticks ; 


class OKCORE_API OpticksMode {
    public:
       static const char* COMPUTE_ARG_ ; 
       static const char* INTEROP_ARG_ ; 
       static const char* NOVIZ_ARG_ ; 
    public:
       static const char* UNSET_MODE_ ;
       static const char* COMPUTE_MODE_ ;
       static const char* INTEROP_MODE_ ;
       static const char* CFG4_MODE_ ;
       enum {
                UNSET_MODE   = 0x1 << 0, 
                COMPUTE_MODE = 0x1 << 1, 
                INTEROP_MODE = 0x1 << 2, 
                CFG4_MODE    = 0x1 << 3
            }; 
    public:
       static unsigned Parse(const char* tag);
    public:
        OpticksMode(const char* tag);  // used to instanciate from OpticksEvent metadata
        OpticksMode(Opticks* ok);
    public:
        int getInteractivityLevel() const ;
        std::string description() const ;
        bool isCompute() const ;
        bool isInterop() const ;
        bool isCfG4() const ;   // needs manual override to set to CFG4_MODE
    public:
        void setOverride(unsigned mode);
    private:
        unsigned  m_mode ;  
        bool      m_compute_requested ;  
        bool      m_noviz ; 
        bool      m_forced_compute ;  
};

#include "OKCORE_TAIL.hh"

