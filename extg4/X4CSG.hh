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

#include <vector>
#include <string>

#include "plog/Severity.h"
#include "X4_API_EXPORT.hh"

class Opticks; 
class G4VSolid ; 

struct nnode ; 
class NPYMeta ; 
class NCSG ; 
class NCSGData ; 
class NCSGList ; 

/**
X4CSG
=======

* Used for testing single G4VSolid in isolation
* Applied to all solids using codegen to construct mains 
  for every solid, see x4gen-

**/

struct X4_API X4CSG 
{
    static const plog::Severity LEVEL ; 
    static const std::string HEAD ; 
    static const std::string TAIL ; 

    static void Serialize( const G4VSolid* solid, const Opticks* ok, const char* csgpath );
    static void GenerateTest( const G4VSolid* solid, const Opticks* ok, const char* prefix, unsigned lvidx );  
    static const char* GenerateTestPath( const char* prefix, unsigned lvidx, const char* ext ) ; 

    static G4VSolid* MakeContainer(const G4VSolid* solid, float scale) ; 
    std::string desc() const ;
    std::string configuration(const char* csgpath) const ;

    X4CSG(const G4VSolid* solid, const Opticks* ok );

    void init();
    void checkTree() const ;
    void setIndex(unsigned index_);

    void configure( NPYMeta* meta );
    void dump(const char* msg="X4CSG::dump");
    std::string save(const char* csgpath) ;
    void loadcheck(const char* csgpath) const ;

    void generateTestMain( std::ostream& out ) const ;
    void dumpTestMain(const char* msg="X4CSG::dumpTestMain") const ;
    void writeTestMain( const char* path ) const ;

    int              verbosity ; 
    const G4VSolid*  solid ; 
    const Opticks*   ok ; 
    std::string      gdml ; 
    const G4VSolid*  container ; 
    const char*      solid_boundary ; 
    const char*      container_boundary ; 
    nnode*           nraw ;   // unbalanced
    nnode*           nsolid ; 
    nnode*           ncontainer ; 
    NCSG*            csolid ; 
    NCSG*            ccontainer ; 
    NCSGList*        ls ; 
    std::vector<NCSG*> trees ;
    int              index ; 

};





