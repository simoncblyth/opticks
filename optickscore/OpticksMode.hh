#pragma once
#include <string>
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksMode {
    public:
       static unsigned int Parse(const char* tag);
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
        OpticksMode(const char* tag);
        OpticksMode(bool compute_requested);
    public:
        void setOverride(unsigned int mode);
        std::string description();
        bool isCompute();
        bool isInterop();
        bool isCfG4();   // needs manual override to set to CFG4_MODE
    private:
         unsigned int m_mode ;  
};

#include "OKCORE_TAIL.hh"

