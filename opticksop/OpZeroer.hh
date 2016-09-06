#pragma once

class OpticksHub ; 

class OPropagator ; 
class OEngineImp ; 
class OContext ; 

// zeroes on GPU record buffer via OptiX or OpenGL

#include "OKOP_API_EXPORT.hh"

class OKOP_API OpZeroer {
   public:
      OpZeroer(OpticksHub* hub, OEngineImp* imp);
   public:
      void setPropagator(OPropagator* propagator);
   public:
      void zeroRecords();
   private:
      void zeroRecordsViaOpenGL();
      void zeroRecordsViaOptiX();
   private:
      OpticksHub*              m_hub ;
      OEngineImp*              m_imp ;
      OContext*                m_ocontext ;
      OPropagator*             m_propagator ;
};

