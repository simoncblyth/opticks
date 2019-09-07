/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
      static const char* buildptxpath_( const char* cu, const char* buildrel, const char* cmake_target) ;
   public:
      OptiXTest(optix::Context& context, const char* cu, const char* raygen_name, const char* exception_name, const char* buildrel, const char* cmake_target);
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


