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


#include <fstream>
#include "BFile.hh"
#include "X4GDMLReadStructure.hh"
#include "X4SolidStore.hh"

#include <cstring>
#include "PLOG.hh"

const plog::Severity X4GDMLReadStructure::LEVEL = PLOG::EnvLevel("X4GDMLReadStructure", "DEBUG" ); 

X4GDMLReadStructure::X4GDMLReadStructure()
{
}

const G4VSolid* X4GDMLReadStructure::read_solid(const char* path, int offset)
{
    G4String fileName = path ; 
    G4bool validation = false ; 
    G4bool isModule = false ; 
    G4bool strip = false ;  

    Read( fileName, validation, isModule, strip ); 

    return X4SolidStore::Get(offset) ;    
}

const G4VSolid* X4GDMLReadStructure::read_solid_from_string(const char* gdmlstring, int offset)
{
    const char* path = WriteGDMLStringToTmpPath(gdmlstring) ;    
    return read_solid(path, offset); 
}


const char* X4GDMLReadStructure::WriteGDMLStringToTmpPath(const char* gdmlstring) // static
{
    const char* pfx = "X4GDMLReadStructure__WriteGDMLStringToTmpPath" ; 
    const char* path = BFile::UserTmpPath(pfx) ; 

    std::ofstream stream(path, std::ios::out); 
    stream.write(gdmlstring, strlen(gdmlstring)) ; 
    stream.close();   
    // curiosly without explicitly closing the stream or closing out the scope 
    // the subsequent read acts as if there was nothing in the file (buffering perhaps?) 
    return path ; 
}

const G4VSolid* X4GDMLReadStructure::ReadSolid(const char* path)
{
    X4GDMLReadStructure reader ; 
    return reader.read_solid(path, -1); 
}

const G4VSolid* X4GDMLReadStructure::ReadSolidFromString(const char* gdmlstring)
{
    X4GDMLReadStructure reader ; 
    return reader.read_solid_from_string(gdmlstring, -1); 
}




