#pragma once

class OpticksHub ; 

class OPropagator ; 
class OContext ; 

class OpEngine ; 

// zeroes on GPU record buffer via OptiX or OpenGL

#include "OKOP_API_EXPORT.hh"

class OKOP_API OpZeroer {
   public:
      OpZeroer(OpticksHub* hub, OpEngine* engine);
   public:
      void setPropagator(OPropagator* propagator);
   public:
      void zeroRecords();
   private:
      void zeroRecordsViaOpenGL();
      void zeroRecordsViaOptiX();
   private:
      OpticksHub*              m_hub ;
      OpEngine*                m_engine ;
      OContext*                m_ocontext ;
      OPropagator*             m_propagator ;
};

