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
            unsigned int addRayGenerationProgram( const char* filename, const char* progname );
            unsigned int addExceptionProgram( const char* filename, const char* progname );
            void setMissProgram( unsigned int index, const char* filename, const char* progname );
     public:
            unsigned int      getNumEntryPoint();
            unsigned int      getNumRayType();
            optix::Context    getContext();
            optix::Group      getTop();
     public:
            static RTformat       getFormat(NPYBase::Type_t type);

            template <typename T>
            static void           upload(optix::Buffer& buffer, NPY<T>* npy);

            template <typename T>
            static void           download(optix::Buffer& buffer, NPY<T>* npy);

            template<typename T>
            optix::Buffer  createIOBuffer(NPY<T>* npy, const char* name, bool set_size=true);   // crucial INTEROP/COMPUTE branch happens here

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


