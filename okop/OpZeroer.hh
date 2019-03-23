#pragma once

class Opticks ; 
class OEvent ; 
class OContext ; 

/**
OpZeroer
==========

Zeroes on GPU record buffer via OptiX or OpenGL

**/

#include "OKOP_API_EXPORT.hh"

class OKOP_API OpZeroer {
   public:
      OpZeroer(Opticks* ok, OEvent* oevt);
   public:
      void zeroRecords();
   private:
      void zeroRecordsViaOpenGL();
      void zeroRecordsViaOptiX();
   private:
      Opticks*                 m_ok ;
      OEvent*                  m_oevt ;
      OContext*                m_ocontext ;
};

