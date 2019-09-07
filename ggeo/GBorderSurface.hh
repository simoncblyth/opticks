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

#include <string>
class GOpticalSurface ; 
#include "GPropertyMap.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/**
GBorderSurface
================




**/

class GGEO_API GBorderSurface : public GPropertyMap<float> {
  public:
      GBorderSurface(const char* name, unsigned int index, GOpticalSurface* optical_surface );
      virtual ~GBorderSurface();
      void Summary(const char* msg="GBorderSurface::Summary", unsigned int imod=1);
      std::string description();
  private:
      void init();
  public:
      void setBorderSurface(const char* pv1, const char* pv2);
      char* getPV1();
      char* getPV2();

  public:
      bool matches(const char* pv1, const char* pv2);
      bool matches_swapped(const char* pv1, const char* pv2);
      bool matches_either(const char* pv1, const char* pv2);
      bool matches_one(const char* pv1, const char* pv2);

  private:
      char* m_bordersurface_pv1 ;  
      char* m_bordersurface_pv2 ;  

};

#include "GGEO_TAIL.hh"

