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
#include <string>
#include "plog/Severity.h"

#include "OpticksCSG.h"
#include <glm/fwd.hpp>

class NCSG ; 
struct nnode ; 

class GCSG ; 
class G4VSolid;
class Opticks ; 

namespace HepGeom
{
   class Transform3D ; 
}

typedef HepGeom::Transform3D G4Transform3D;

/**
CMaker
======

CMaker is a constitent of CTestDetector used
to convert NCSG/nnode shapes into G4VSolid. 
CMaker::makeSolid handles booleans via recursive calls.

An earlier implementation from prior to NCSG/nnode is in CMaker.cc.old 


**/

#include "CFG4_API_EXPORT.hh"

class CFG4_API CMaker 
{
    public:
        static const plog::Severity LEVEL ; 
    public:
        static std::string PVName(const char* shapename, int idx=-1);
        static std::string LVName(const char* shapename, int idx=-1);
    public:
        CMaker();
    public:
        static G4VSolid* MakeSolid(const NCSG* csg);                  
        static G4VSolid* MakeSolid(const nnode* node);
        static G4Transform3D* ConvertTransform(const glm::mat4& t);
    private:
        static G4VSolid* MakeSolid_r(const nnode* node, unsigned depth );
        static G4VSolid*      ConvertPrimitive(const nnode* node);
        static G4VSolid*      ConvertZSphere(const nnode* node);
        static G4VSolid*      ConvertOperator( const nnode* node, G4VSolid* left, G4VSolid* right, unsigned depth ); 
        static G4VSolid*      ConvertConvexPolyhedron(const nnode* node);

};


