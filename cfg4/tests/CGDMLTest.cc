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

#include "OPTICKS_LOG.hh"

#include "NMeta.hpp"
#include "G4GDMLParser.hh"
#include "G4VPhysicalVolume.hh"

#include "CGDML.hh"


struct Demo 
{
   int answer ; 
};


void test_GenerateName()
{
    Demo* d = new Demo { 42 } ;   
    LOG(info) << CGDML::GenerateName( "Demo", d, true );  
}

/**

::

   cp $(opticksaux-dx1) /tmp/v1.gdml

**/


void test_load_0(const char* path)
{
    LOG(info) << path ; 
    bool validate = false ; 
    bool trimPtr = false ; 
    G4GDMLParser parser;
    parser.SetStripFlag(trimPtr);
    parser.Read(path, validate);
    G4VPhysicalVolume* world =  parser.GetWorldVolume() ;
    assert( world ); 
}

void test_load(const char* path)
{
    CGDML cg(path); 
    G4VPhysicalVolume* world =  cg.getWorldVolume() ;
    assert( world ); 

    cg.dumpAux("test_load.dumpAux"); 

    NMeta* m = cg.getAuxMeta(); 
    m->dump("test_load.getAuxMeta"); 
}

void test_Parse(const char* path)
{
    NMeta* meta = NULL ;  
    G4VPhysicalVolume* world = CGDML::Parse(path, &meta); 
    assert( world ); 
    if(meta)
    {
        meta->dump("test_Parse"); 
    }
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* path = argc > 1 ? argv[1] : NULL ; 
    if(!path) LOG(error) << " expecting path to GDML " ; 
    if(!path) return 0 ; 

    //test_GenerateName(); 
    //test_load(path); 
    test_Parse(path); 

    return 0 ; 
}
// om-;TEST=CGDMLTest om-t
