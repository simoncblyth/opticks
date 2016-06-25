#pragma once

class OpticksEvent ; 
class OPropagator ; 
class OContext ; 

// zeroes on GPU record buffer via OptiX or OpenGL

#include "OKOP_API_EXPORT.hh"

class OKOP_API OpZeroer {
   public:
      OpZeroer(OContext* ocontext);
   public:
      void setEvent(OpticksEvent* evt);
      void setPropagator(OPropagator* propagator);
   public:
      void zeroRecords();
   private:
      void zeroRecordsViaOpenGL();
      void zeroRecordsViaOptiX();
   private:
      OContext*                m_ocontext ;
      OpticksEvent*                m_evt ;
      OPropagator*             m_propagator ;
};

