#pragma once

/**
OContext
=========

Wrapper for OptiX context providing numerous utilities including: 

* upload/download using NPY arrays
* program construction
* output redirection 
* snapping PPM image of output buffer
* saving output buffer into NPY array

**/


#include <string>
#include "OXPPNS.hh"
#include "plog/Severity.h"

#include "NPYBase.hpp"
template <typename T> class NPY ; 
class OConfig ; 
class OpticksEntry ; 
class Opticks ; 
class BTimes ; 
//struct STimes ; 

#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OContext {

        // access to lowlevel  addEntry 
        friend class OLaunchTest ; 
        friend class LTOOContextUploadDownloadTest ;
        friend class OAxisTest ;
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
        static plog::Severity LEVEL ; 

     public:
            static const char* LaunchLogPath(unsigned index); 
            const char* getPrintIndexLogPath() const  ; 
     public:
            OContext(optix::Context context, Opticks* ok, bool with_top=true, bool verbose=false, const char* cmake_target="OptiXRap");
            void cleanUp();
     public:
            const char* getModeName();
            OContext::Mode_t getMode();
            bool isCompute();
            bool isInterop();
            void snap(const char* path="/tmp/snap.ppm");
            void save(const char* path="/tmp/snap.npy");
     private:
            void init();
            void initPrint();
     public:
            void launch(unsigned lmode, unsigned entry, unsigned width, unsigned height=1, BTimes* times=NULL);
     private:
            double validate_();
            double compile_();
            double launch_(unsigned entry, unsigned width, unsigned height=1 );
            double launch_redirected_(unsigned entry, unsigned width, unsigned height=1 );
     public:
            // pass thru to OConfig
            optix::Program createProgram(const char* cu_filename, const char* progname );
            void dump(const char* msg="OContext::dump");
            void close();
     public:
            OpticksEntry*  addEntry(char code='G');
            unsigned int   addEntry(const char* cu_filename="generate.cu", const char* raygen="generate", const char* exception="exception", bool defer=true);
            void setMissProgram( unsigned int index, const char* filename, const char* progname, bool defer=true);
     private:
            unsigned int addRayGenerationProgram( const char* filename, const char* progname, bool defer=true);
            unsigned int addExceptionProgram( const char* filename, const char* progname, bool defer=true);
     public:
            unsigned          getNumEntryPoint();
            unsigned          getNumRayType();
            unsigned          getDebugPhoton() const ;
            optix::Context&   getContextRef();
            optix::Context    getContext();
            optix::Group      getTop();
     public:
            static RTformat       getFormat(NPYBase::Type_t type, bool seed);

            template <typename T>
            static void           upload(optix::Buffer& buffer, NPY<T>* npy);

            template <typename T>
            static void           download(optix::Buffer& buffer, NPY<T>* npy);

     public:
            optix::Buffer createEmptyBufferF4() ;

            template<typename T>
            optix::Buffer  createBuffer(NPY<T>* npy, const char* name);  
     private:
            template<typename T>
            void configureBuffer(optix::Buffer& buffer, NPY<T>* npy, const char* name);  

            template <typename T>
            static unsigned determineBufferSize(NPY<T>* npy, const char* name);
      public:
            template<typename T>
            static void resizeBuffer(optix::Buffer& buffer, NPY<T>* npy, const char* name);  
     private:
            optix::Context    m_context ; 
            optix::Group      m_top ; 
            Opticks*          m_ok ; 
            OConfig*          m_cfg ; 
            Mode_t            m_mode ; 
            int               m_debug_photon ; 
            unsigned          m_entry ; 
            bool              m_closed ; 
            bool              m_with_top ; 
            bool              m_verbose ; 
            const char*       m_cmake_target ; 
            const char*       m_llogpath ; 
            unsigned          m_launch_count ; 
};


