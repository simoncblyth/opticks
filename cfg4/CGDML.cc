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


#include "SGDML.hh"
#include "CGDML.hh"
#include "BFile.hh"
#include "G4GDMLParser.hh"

#include "PLOG.hh"




G4VPhysicalVolume* CGDML::Parse(const char* path) // static 
{
    if( path == NULL )
    {   
        LOG(fatal) << " path to an existing gdml file is required " ; 
        return NULL  ; 
    }   

    LOG(info) << "path " << path ; 
    bool validate = false ; 
    bool trimPtr = false ; 
    G4GDMLParser parser;
    parser.SetStripFlag(trimPtr);
    parser.Read(path, validate);
    return parser.GetWorldVolume() ;
}


void CGDML::Export(const char* dir, const char* name, const G4VPhysicalVolume* const world )
{
    std::string path = BFile::FormPath(dir, name);
    CGDML::Export( path.c_str(), world ); 
}

void CGDML::Export(const char* path, const G4VPhysicalVolume* const world )
{
    assert( world );

    bool exists = BFile::ExistsFile( path ); 

    // cannot skip and reuse existing despite it having the same digest 
    // as the pointer locations will differ so all the names will be different
    // relative to those in lv2sd for example
    if(exists) 
    {
        BFile::RemoveFile( path ) ; 
    }

    bool create = true ; 
    BFile::preparePath( path, create ) ;   

    LOG(info) << "export to " << path ; 

    G4GDMLParser* gdml = new G4GDMLParser ;
    G4bool refs = true ;
    G4String schemaLocation = "" ; 

    gdml->Write(path, world, refs, schemaLocation );
}


// based on G4GDMLWrite::GenerateName 
std::string CGDML::GenerateName(const char* name, const void* const ptr, bool addPointerToName )
{
    return SGDML::GenerateName(name, ptr, addPointerToName );
}



