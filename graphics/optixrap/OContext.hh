#pragma once

#include <optixu/optixpp_namespace.h>

class OContext {
    public:
        enum { 
               e_pinhole_camera_entry,
               e_generate_entry,
               e_entryPointCount 
            };

        enum {
                e_radiance_ray,
                e_touch_ray,
                e_propagate_ray,
                e_rayTypeCount 
             };

     public:
            OContext(optix::Context context);
     public:
            unsigned int      getNumEntryPoint();
            unsigned int      getNumRayType();
            optix::Context    getContext();
            optix::Group      getTop();
     private:
            void init();
     private:
            optix::Context    m_context ; 
            optix::Group      m_top ; 

};


inline OContext::OContext(optix::Context context) : m_context(context)
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
inline unsigned int OContext::getNumEntryPoint()
{
    //return m_evt ? e_entryPointCount : e_entryPointCount - 1 ; 
    return e_entryPointCount ; 
}






