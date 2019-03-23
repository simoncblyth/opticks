#pragma once

/**
OptiXTest
===========

Used for standalone-ish OptiX tests providing minimal context setup.

Usage examples: 

* optixrap/tests/OOMinimalTest.cc
* optixrap/tests/intersect_analytic_test.cc


**/


#include "OXRAP_PUSH.hh"
#include <optixu/optixpp_namespace.h>
#include "OXRAP_POP.hh"

#include <string>
#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OptiXTest {
   public:
      static std::string ptxname_(const char* projname, const char* name);
      static const char* ptxpath_( const char* cu, const char* projdir="optixrap", const char* projname="OptiXRap") ;
   public:
      OptiXTest(optix::Context& context, const char* cu, const char* raygen_name, const char* exception_name="exception"); 
      std::string description();
      void Summary(const char* msg="OptiXTest::Summary");
   private:
      void init(optix::Context& context);
   private:
      const char* m_cu ; 
      const char* m_ptxpath ; 
      const char* m_raygen_name ; 
      const char* m_exception_name ; 

};


