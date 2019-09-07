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

#include "GPropertyMap.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GSource : public GPropertyMap<float> {
  public:
      static GSource* make_blackbody_source(const char* name, unsigned int index, float kelvin=6500.f);
  public:
      GSource(GSource* other);
      GSource(const char* name, unsigned int index);
      virtual ~GSource();
  private:
      void init(); 
  public: 
      void Summary(const char* msg="GSource::Summary");

};

#include "GGEO_TAIL.hh"




