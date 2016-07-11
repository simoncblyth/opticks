#pragma once

class OContext ; 
template <typename T> class NPY ; 

#include "OXRAP_PUSH.hh"
#include <optixu/optixpp_namespace.h>
#include "OXRAP_POP.hh"

#include "OKGL_API_EXPORT.hh"
#include "SLauncher.hh"

class OKGL_API OAxisTest : public SLauncher {
    public:
        OAxisTest(OContext* ocontext, NPY<float>* axis_data);
        void prelaunch();
        void download();
    public:
        virtual void launch(unsigned count);
    private:
        void init();
    private:
        OContext*       m_ocontext ;
        NPY<float>*     m_axis_data ; 
        optix::Buffer   m_buffer ; 
        unsigned        m_ni ; 
        unsigned        m_entry ; 
};


