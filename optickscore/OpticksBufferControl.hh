#pragma once
#include <string>
#include <vector>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksBufferControl {
    public:
        enum {
                OPTIX_SETSIZE        = 0x1 << 1,
                OPTIX_NON_INTEROP    = 0x1 << 2,
                OPTIX_INPUT_OUTPUT   = 0x1 << 3,
                OPTIX_INPUT_ONLY     = 0x1 << 4,
                OPTIX_OUTPUT_ONLY    = 0x1 << 5,
                PTR_FROM_OPTIX       = 0x1 << 6,
                PTR_FROM_OPENGL      = 0x1 << 7,
                UPLOAD_WITH_CUDA     = 0x1 << 8,
                DUMMY                = 0x1 << 31 
             };  

        static std::string Description(unsigned long long ctrl);
        static unsigned long long Parse(const char* ctrl, char delim=',');
        static unsigned long long ParseTag(const char* ctrl);
        static bool isSet(unsigned long long ctrl, const char* mask);
        static std::vector<const char*> Tags();
    private:
        static const char* OPTIX_SETSIZE_ ; 
        static const char* OPTIX_NON_INTEROP_ ; 
        static const char* OPTIX_INPUT_OUTPUT_ ; 
        static const char* OPTIX_INPUT_ONLY_ ; 
        static const char* OPTIX_OUTPUT_ONLY_ ; 
        static const char* PTR_FROM_OPTIX_ ; 
        static const char* PTR_FROM_OPENGL_ ; 
        static const char* UPLOAD_WITH_CUDA_ ; 

    public:
        OpticksBufferControl(const char* ctrl); 
        OpticksBufferControl(unsigned long long ctrl); 
        bool isSet(const char* mask) const;
    private:
         unsigned long long m_ctrl ; 

};

#include "OKCORE_TAIL.hh"


