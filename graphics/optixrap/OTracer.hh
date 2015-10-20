#pragma once

#include <optixu/optixpp_namespace.h>

class OContext ; 
class Composition ; 
struct OTimes ; 

class OTracer {
   public:
       OTracer(OContext* ocontext, Composition* composition);
   public:
       void trace();
       void report(const char* msg="OTracer::report");
       void setResolutionScale(unsigned int resolution_scale);
       unsigned int getResolutionScale();
       unsigned int getTraceCount();
   private:
       void init();

   private:
       OContext*       m_ocontext ; 
       Composition*    m_composition ; 
       optix::Context  m_context ; 
       unsigned int    m_resolution_scale ; 

       OTimes*          m_trace_times ; 
       unsigned int     m_trace_count ; 
       double           m_trace_prep ; 
       double           m_trace_time ; 

};


inline OTracer::OTracer(OContext* ocontext, Composition* composition) :
    m_ocontext(ocontext),
    m_composition(composition),
    m_trace_times(NULL),
    m_trace_count(0),
    m_trace_prep(0),
    m_trace_time(0)
{
    init();
}


inline void OTracer::setResolutionScale(unsigned int resolution_scale)
{
    m_resolution_scale = resolution_scale ; 
}
inline unsigned int OTracer::getResolutionScale()
{
    return m_resolution_scale ; 
}
inline unsigned int OTracer::getTraceCount()
{
    return m_trace_count ; 
}





