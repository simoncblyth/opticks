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
#include <vector>

typedef enum {
    // generated by x4-enum- Mon Jun 11 20:10:02 HKT 2018 
    _G4DisplacedSolid ,
    _G4UnionSolid ,
    _G4IntersectionSolid ,
    _G4SubtractionSolid ,
    _G4MultiUnion ,
    _G4Box ,
    _G4Cons ,
    _G4EllipticalCone ,
    _G4Ellipsoid ,
    _G4EllipticalTube ,
    _G4ExtrudedSolid ,
    _G4Hype ,
    _G4Orb ,
    _G4Para ,
    _G4Paraboloid ,
    _G4Polycone ,
    _G4GenericPolycone ,
    _G4Polyhedra ,
    _G4Sphere ,
    _G4TessellatedSolid ,
    _G4Tet ,
    _G4Torus ,
    _G4GenericTrap ,
    _G4Trap ,
    _G4Trd ,
    _G4Tubs ,
    _G4CutTubs ,
    _G4TwistedBox ,
    _G4TwistedTrap ,
    _G4TwistedTrd ,
    _G4TwistedTubs 
} X4Entity_t ;

#include "X4_API_EXPORT.hh"

class X4_API X4Entity
{
    public:
        static X4Entity_t   Type(const char* name); 
        static const char*  Name(X4Entity_t type); 
    private:
        X4Entity();
        X4Entity_t type(const char* name);
        const char* name(X4Entity_t type); 

        typedef std::vector<std::string> VN ; 
        typedef std::vector<X4Entity_t>  VT ; 

        VN m_names ; 
        VT m_types ; 

        static X4Entity* fEntity ; 

}; 
