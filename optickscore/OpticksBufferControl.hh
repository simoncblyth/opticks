#pragma once
#include <string>
#include <vector>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksBufferControl {
    public:
        enum {
                OPTIX_SETSIZE           = 0x1 << 1,
                OPTIX_NON_INTEROP       = 0x1 << 2,
                OPTIX_INPUT_OUTPUT      = 0x1 << 3,
                OPTIX_INPUT_ONLY        = 0x1 << 4,
                OPTIX_OUTPUT_ONLY       = 0x1 << 5,
                PTR_FROM_OPTIX          = 0x1 << 6,
                PTR_FROM_OPENGL         = 0x1 << 7,
                UPLOAD_WITH_CUDA        = 0x1 << 8,
                BUFFER_COPY_ON_DIRTY    = 0x1 << 9,
                BUFFER_GPU_LOCAL        = 0x1 << 10,
                INTEROP_MODE            = 0x1 << 11,
                COMPUTE_MODE            = 0x1 << 12,
                DUMMY                   = 0x1 << 31 
             };  
    public:
        static const char* OPTIX_SETSIZE_ ; 
        static const char* OPTIX_NON_INTEROP_ ; 
        static const char* OPTIX_INPUT_OUTPUT_ ; 
        static const char* OPTIX_INPUT_ONLY_ ; 
        static const char* OPTIX_OUTPUT_ONLY_ ; 
        static const char* PTR_FROM_OPTIX_ ; 
        static const char* PTR_FROM_OPENGL_ ; 
        static const char* UPLOAD_WITH_CUDA_ ; 
        static const char* BUFFER_COPY_ON_DIRTY_ ; 
        static const char* BUFFER_GPU_LOCAL_ ; 
        static const char* INTEROP_MODE_ ; 
        static const char* COMPUTE_MODE_ ; 

    public:
        static std::string Description(unsigned long long ctrl);
        static unsigned long long Parse(const char* ctrl, char delim=',');
        static unsigned long long ParseTag(const char* ctrl);
        static bool isSet(unsigned long long ctrl, const char* mask);
        static std::vector<const char*> Tags();
    public:
     //   OpticksBufferControl(const char* ctrl); 
        OpticksBufferControl(unsigned long long* ctrl); 
        void add(const char* mask);
        bool isSet(const char* mask) const;
        bool operator()(const char* mask);

        std::string description(const char* msg="OpticksBufferControl::description") const;
    private:
         unsigned long long* m_ctrl ; 

};

#include "OKCORE_TAIL.hh"


