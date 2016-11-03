#pragma once

class Opticks ; 
class OContext ; 

#include "OXPPNS.hh"
class cuRANDWrapper ; 

#include "OXRAP_API_EXPORT.hh"
class OXRAP_API ORng 
{
   public:
      ORng(Opticks* ok, OContext* ocontext);
   private:
      void init(); 
   private:
      Opticks*        m_ok ; 
      OContext*       m_ocontext ; 
      optix::Context  m_context ;
    protected:
      optix::Buffer   m_rng_states ;
      cuRANDWrapper*  m_rng_wrapper ;


};
