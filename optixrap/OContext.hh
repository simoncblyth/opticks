#pragma once

#include <string>
#include "OXPPNS.hh"

#include "NPYBase.hpp"
template <typename T> class NPY ; 
class OConfig ; 
struct OTimes ; 

#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OContext {
    public:
        enum {
                e_propagate_ray,
                e_radiance_ray,
                e_touch_ray,
                e_rayTypeCount 
             };

        enum {
                VALIDATE  = 0x1 << 1, 
                COMPILE   = 0x1 << 2,
                PRELAUNCH = 0x1 << 3,
                LAUNCH    = 0x1 << 4
             };

        enum {
                BUFOPT_SETSIZE        = 0x1 << 1,
                BUFOPT_NON_INTEROP    = 0x1 << 2,
                BUFOPT_INPUT_OUTPUT   = 0x1 << 3,
                BUFOPT_INPUT_ONLY     = 0x1 << 4,
                BUFOPT_OUTPUT_ONLY    = 0x1 << 5
             };

        static std::string BufOptString(unsigned int bufopt);
        static const char* BUFOPT_SETSIZE_ ; 
        static const char* BUFOPT_NON_INTEROP_ ; 
        static const char* BUFOPT_INPUT_OUTPUT_ ; 
        static const char* BUFOPT_INPUT_ONLY_ ; 
        static const char* BUFOPT_OUTPUT_ONLY_ ; 

        typedef enum { COMPUTE, INTEROP } Mode_t ;   
        static const char* COMPUTE_ ; 
        static const char* INTEROP_ ; 

     public:
            OContext(optix::Context context, Mode_t mode, bool with_top=true);
            void cleanUp();
     public:
            const char* getModeName();
            OContext::Mode_t getMode();
            bool isCompute();
            bool isInterop();
     public:
            void setStackSize(unsigned int stacksize);
            void setPrintIndex(const std::string& pindex);
            void setDebugPhoton(unsigned int debug_photon);
            unsigned int getDebugPhoton();
     public:
            void launch(unsigned int lmode, unsigned int entry, unsigned int width, unsigned int height=1, OTimes* times=NULL);
     public:
            // pass thru to OConfig
            optix::Program createProgram(const char* filename, const char* progname );
            void dump(const char* msg="OContext::dump");
            void close();
     public:
            unsigned int addEntry(const char* filename="generate.cu.ptx", const char* raygen="generate", const char* exception="exception", bool defer=true);
            void setMissProgram( unsigned int index, const char* filename, const char* progname, bool defer=true);
     private:
            unsigned int addRayGenerationProgram( const char* filename, const char* progname, bool defer=true);
            unsigned int addExceptionProgram( const char* filename, const char* progname, bool defer=true);
     public:
            unsigned int      getNumEntryPoint();
            unsigned int      getNumRayType();
            optix::Context&   getContextRef();
            optix::Context    getContext();
            optix::Group      getTop();
     public:
            static RTformat       getFormat(NPYBase::Type_t type);

            template <typename T>
            static void           upload(optix::Buffer& buffer, NPY<T>* npy);

            template <typename T>
            static void           download(optix::Buffer& buffer, NPY<T>* npy);

     public:
            template<typename T>
            optix::Buffer  createBuffer(NPY<T>* npy, const char* name, unsigned int bufopt=BUFOPT_SETSIZE | BUFOPT_INPUT_OUTPUT );  
     private:
            template<typename T>
            void configureBuffer(optix::Buffer& buffer, NPY<T>* npy, const char* name, unsigned int bufopt);  
     private:
            void init();
     private:
            optix::Context    m_context ; 
            optix::Group      m_top ; 
            OConfig*          m_cfg ; 
            Mode_t            m_mode ; 
            int               m_debug_photon ; 
            unsigned int      m_entry ; 
            bool              m_closed ; 
            bool              m_with_top ; 
};


