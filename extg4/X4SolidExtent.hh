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

#include <glm/fwd.hpp>

class G4VSolid ; 
struct nbbox ; 

#include "G4Transform3D.hh"
#include "X4_API_EXPORT.hh"

/**
X4SolidExtent
===============

Duplicate of CFG4.CSolid as 10.4.2 is crashing in G4VisExtent

**/

class X4_API X4SolidExtent {

   public:
       static nbbox* Extent( const G4VSolid* solid ); 
   public:
       X4SolidExtent(const G4VSolid* solid);
       void extent(const G4Transform3D& tran, glm::vec3& low, glm::vec3& high, glm::vec4& center_extent);
   private:
       const G4VSolid* m_solid ; 
};


