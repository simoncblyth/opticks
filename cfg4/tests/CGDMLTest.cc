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

#include "BMeta.hh"
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



struct CGDMLTest 
{
    static void test_load(const char* path); 
    static void test_Parse(const char* path); 
    static void test_Parse_Export(const char* ipath, const char* opath); 
}; 



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

void CGDMLTest::test_load(const char* path)
{
    CGDML cg ;
    cg.read(path); 
    G4VPhysicalVolume* world =  cg.getWorldVolume() ;
    assert( world ); 

    cg.dumpLVMeta("test_load.dumpLVMeta"); 
    cg.dumpUserMeta("test_load.dumpUserMeta"); 

    BMeta* m = cg.getLVMeta(); 
    m->dump("test_load.getLVMeta"); 
}

void CGDMLTest::test_Parse(const char* path)
{
    BMeta* meta = NULL ;  
    G4VPhysicalVolume* world = CGDML::Parse(path, &meta); 
    assert( world ); 
    if(meta) meta->dump("CGDMLTest::test_Parse"); 
}


/**
Parse and then Export by default looses the GDMLAux metadata.
**/

void CGDMLTest::test_Parse_Export(const char* ipath, const char* opath)
{
    BMeta* meta = NULL ;  
    G4VPhysicalVolume* world = CGDML::Parse(ipath, &meta); 
    assert( world ); 
    if(meta) meta->dump("CGDMLTest::test_Parse_Export"); 
    if(opath) CGDML::Export(opath, world, meta); 
}

/**

::

   opticksaux-;cp $(opticksaux-dx1) /tmp/v1.gdml

**/


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* ipath = argc > 1 ? argv[1] : NULL ; 
    const char* opath = argc > 2 ? argv[2] : NULL ; 
    if(!ipath) LOG(error) << " expecting path to GDML " ; 
    if(!ipath) return 0 ; 

    //test_GenerateName(); 
    //CGDMLTest::test_load(ipath); 
    //CGDMLTest::test_Parse(ipath); 
    CGDMLTest::test_Parse_Export(ipath, opath); 

    return 0 ; 
}
// om-;TEST=CGDMLTest om-t
