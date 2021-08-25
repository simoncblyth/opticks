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
#include "plog/Severity.h"
#include <string>

class G4VSolid ; 

/**
X4GDMLWrite
=============

Subclass of the standard Geant4 G4GDMLWriteStructure
that adds methods for writing single solids to GDML snippets
either in files or strings.

g4-;g4-cls G4GDMLWrite
g4-;g4-cls G4GDMLWriteStructure

Note that the G4 GDML implementation uses
a bizarre long inheritance chain : so you just have to 
resort to searching for methods.

**/


#include "G4GDMLWriteStructure.hh"

class X4_API X4GDMLWriteStructure : public G4GDMLWriteStructure 
{
    public:
        static const plog::Severity LEVEL ; 
    public:
        X4GDMLWriteStructure(bool refs) ; 

        void write(const G4VSolid* solid, const char* path=NULL ); // to file or stdout when path is NULL
        std::string to_string( const G4VSolid* solid ); 
   private:
        void init(bool refs);
        void add( const G4VSolid* solid ); 
        std::string write( const char* path=NULL ) ; 
  
   private:
        xercesc::DOMElement* gdml ; 
        xercesc::DOMImplementation* impl ;

        XMLCh local_tempStr[10000]  ;

};

