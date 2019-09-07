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

#include "X4_API_EXPORT.hh"
#include <string>

class G4VSolid ;
class X4GDMLWriteStructure ;  

/**
X4GDMLParser
=============

g4-;g4-cls G4GDMLParser

**/

#include "G4String.hh"
#include "G4GDMLParser.hh"

class X4_API X4GDMLParser  
{
    public:
        static const char* PreparePath( const char* prefix, int lvidx, const char* ext=".gdml"  ); 
    public:
        static void Write( const G4VSolid* solid, const char* path, bool refs );  // NULL path writes to stdout
        static std::string ToString( const G4VSolid* solid, bool refs ) ;  
    private:
        X4GDMLParser(bool refs) ; 
        void write(const G4VSolid* solid, const char* path);
        void write_noisily(const G4VSolid* solid, const char* path);
        std::string to_string( const G4VSolid* solid);
    private:
        X4GDMLWriteStructure* writer ;  

};

