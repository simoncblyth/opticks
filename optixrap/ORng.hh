#pragma once

class Opticks ; 
class OContext ; 

#include <vector>
#include "OXPPNS.hh"
#include "plog/Severity.h"
class cuRANDWrapper ; 

#include "OXRAP_API_EXPORT.hh"

/**
ORng
====

Uploads persisted curand rng_states to GPU.
Canonical instance m_orng is ctor resident of OPropagator.

Work is mainly done by cudarap-/cuRANDWrapper

TODO: investigate Thrust based alternatives for curand initialization 
      potential for eliminating cudawrap- 

**/


class OXRAP_API ORng 
{
   public:
      static const plog::Severity LEVEL ; 
   public:
      ORng(Opticks* ok, OContext* ocontext);
   private:
      void init(); 
   private:
      Opticks*        m_ok ; 
      const std::vector<unsigned>& m_mask ; 
      OContext*       m_ocontext ; 
      optix::Context  m_context ;
    protected:
      optix::Buffer   m_rng_states ;
      cuRANDWrapper*  m_rng_wrapper ;


};
