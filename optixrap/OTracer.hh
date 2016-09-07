#pragma once

#include "OXPPNS.hh"

class OContext ; 
class Composition ; 
struct STimes ; 

#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OTracer {
   public:
       OTracer(OContext* ocontext, Composition* composition);
   public:
       void trace_();
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

       STimes*          m_trace_times ; 
       unsigned int     m_trace_count ; 
       double           m_trace_prep ; 
       double           m_trace_time ; 

       int              m_entry_index ; 

};




