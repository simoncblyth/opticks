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
class G4OpticalSurface ; 

#include "CFG4_API_EXPORT.hh"
class CFG4_API COpticalSurface 
{
  public:
      static std::string Brief(G4OpticalSurface* os);
  public:
      static const char* dielectric_dielectric_ ;
      static const char* dielectric_metal_      ;
      static const char* Type(G4SurfaceType type);
  public:
      static const char* polished_ ;
      static const char* polishedfrontpainted_ ;
      static const char* polishedbackpainted_  ;
      static const char* ground_ ;
      static const char* groundfrontpainted_ ;
      static const char* groundbackpainted_  ;
      static const char* Finish( G4OpticalSurfaceFinish finish);
  public:
      static const char* glisur_ ;
      static const char* unified_;
      static const char* Model( G4OpticalSurfaceModel model );
  public:
      COpticalSurface(G4OpticalSurface* os);
      std::string brief();
  private:
      G4OpticalSurface* m_os ; 

};
