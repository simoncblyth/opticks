#pragma once

class OConfig ; 
struct OTimes ; 
#include <string>
#include <optixu/optixpp_namespace.h>
#include "NPY.hpp"

//  problem in tracer mode is that context doesnt 
//  compile when no generate program is attached
//  dumb fix is to flip enum ordering, need better way

class OContext {
    public:

      /*
        enum { 
               e_generate_entry,
               e_pinhole_camera_entry,
               e_entryPointCount 
            };
       */

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
            OContext(optix::Context context, Mode_t mode);
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
            optix::Buffer  createIOBuffer(NPY<T>* npy, const char* name);

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
           

};


inline OContext::OContext(optix::Context context, Mode_t mode) 
    : 
    m_context(context),
    m_mode(mode),
    m_debug_photon(-1),
    m_entry(0),
    m_closed(false)
{
    init();
}

inline optix::Context OContext::getContext()
{
     return m_context ; 
}
inline optix::Group OContext::getTop()
{
     return m_top ; 
}
inline unsigned int OContext::getNumRayType()
{
    return e_rayTypeCount ;
}


inline void OContext::setDebugPhoton(unsigned int debug_photon)
{
    m_debug_photon = debug_photon ; 
}
inline unsigned int OContext::getDebugPhoton()
{
    return m_debug_photon ; 
}


inline OContext::Mode_t OContext::getMode()
{
    return m_mode ; 
}


inline bool OContext::isCompute()
{
    return m_mode == COMPUTE ; 
}
inline bool OContext::isInterop()
{
    return m_mode == INTEROP ; 
}



