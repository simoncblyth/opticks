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
struct guint4 ; 

#include "GGEO_API_EXPORT.hh"

class GGEO_API GOpticalSurface {
  public:
      // type related
      static const char* dielectric_dielectric_ ;
      static const char* dielectric_metal_      ;
      static const char* Type(unsigned type);

      const char* getType() const ;
  public:
      // finish related
      static const char* polished_ ;
      static const char* polishedfrontpainted_ ;
      static const char* polishedbackpainted_  ;
      static const char* ground_ ;
      static const char* groundfrontpainted_ ;
      static const char* groundbackpainted_  ;
      static const char* Finish(unsigned finish);
      static bool IsPolished(unsigned finish); 
      static bool IsGround(  unsigned finish); 

      const char* getFinish() const ;
      int getFinishInt() const ; 
      bool isPolished() const ;
      bool isGround() const ;
      bool isSpecular() const ;
      bool isSpecularOld() const ;
  public:
      static std::string brief(const guint4& optical); 
  public:
      static GOpticalSurface* create(const char* name, guint4 opt );
      GOpticalSurface(GOpticalSurface* other);
      GOpticalSurface(const char* name, const char* type, const char* model, const char* finish, const char* value);
  private:
      void init();
      void findShortName(char marker='_');
      void checkValue() const ;
  public:
      virtual ~GOpticalSurface();

      guint4 getOptical() const ;
      
      std::string description() const ;
      void Summary(const char* msg="GOpticalSurface::Summary", unsigned int imod=1) const ;
      const char* digest() const ;
  public:
      const char* getName() const ;
      const char* getModel() const ;
      const char* getValue() const ;
      const char* getShortName() const ;
  private:
      const char* m_name ;  
      const char* m_type ;  
      const char* m_model ;  
      const char* m_finish ;  
      const char* m_value ;  
      const char* m_shortname ;  

};


