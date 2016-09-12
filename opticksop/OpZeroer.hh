#pragma once

class OpticksHub ; 
class OEvent ; 
class OContext ; 


// zeroes on GPU record buffer via OptiX or OpenGL

#include "OKOP_API_EXPORT.hh"

class OKOP_API OpZeroer {
   public:
      OpZeroer(OpticksHub* hub, OEvent* oevt);
   public:
      void zeroRecords();
   private:
      void zeroRecordsViaOpenGL();
      void zeroRecordsViaOptiX();
   private:
      OpticksHub*              m_hub ;
      OEvent*                  m_oevt ;
      OContext*                m_ocontext ;
};

