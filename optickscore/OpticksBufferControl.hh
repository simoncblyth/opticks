#pragma once
#include <string>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksBufferControl {
    public:
        enum {
                OPTIX_SETSIZE        = 0x1 << 1,
                OPTIX_NON_INTEROP    = 0x1 << 2,
                OPTIX_INPUT_OUTPUT   = 0x1 << 3,
                OPTIX_INPUT_ONLY     = 0x1 << 4,
                OPTIX_OUTPUT_ONLY    = 0x1 << 5
             };  

        static std::string Description(unsigned long long ctrl);
        static unsigned long long Parse(const char* ctrl, char delim=',');
        static unsigned long long ParseTag(const char* ctrl);
    private:
        static const char* OPTIX_SETSIZE_ ; 
        static const char* OPTIX_NON_INTEROP_ ; 
        static const char* OPTIX_INPUT_OUTPUT_ ; 
        static const char* OPTIX_INPUT_ONLY_ ; 
        static const char* OPTIX_OUTPUT_ONLY_ ; 

};

#include "OKCORE_TAIL.hh"


