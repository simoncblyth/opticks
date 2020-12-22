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


#include <map>
#include <fstream>
#include <cstring>

#include "BFile.hh"
#include "X4GDMLReadStructure.hh"
#include "X4SolidStore.hh"
#include "X4GDMLMatrix.hh"

#include "G4String.hh"
#include "G4GDMLReadDefine.hh"  // for G4GDMLMatrix

#include "PLOG.hh"

const plog::Severity X4GDMLReadStructure::LEVEL = PLOG::EnvLevel("X4GDMLReadStructure", "DEBUG" ); 

X4GDMLReadStructure::X4GDMLReadStructure()
{
}

void X4GDMLReadStructure::readString(const char* gdmlstring)
{
    const char* path = WriteGDMLStringToTmpPath(gdmlstring) ;    
    readFile(path); 
}

void X4GDMLReadStructure::readFile(const char* path)
{
    G4String fileName = path ; 
    G4bool validation = false ; 
    G4bool isModule = false ; 
    G4bool strip = false ;  

    Read( fileName, validation, isModule, strip ); 
}


const G4VSolid* X4GDMLReadStructure::getSolid(int offset)
{
    return X4SolidStore::Get(offset) ;    
}


void X4GDMLReadStructure::WriteGDMLString(const char* gdmlstring, const char* path) // static
{
    std::ofstream stream(path, std::ios::out); 
    stream.write(gdmlstring, strlen(gdmlstring)) ; 
    stream.close();   
    // curiosly without explicitly closing the stream or closing out the scope 
    // the subsequent read acts as if there was nothing in the file (buffering perhaps?) 
}

const char* X4GDMLReadStructure::WriteGDMLStringToTmpPath(const char* gdmlstring) // static
{
    const char* pfx = "X4GDMLReadStructure__WriteGDMLStringToTmpPath" ; 
    const char* path = BFile::UserTmpPath(pfx) ; 
    WriteGDMLString(gdmlstring, path); 
    return path ; 
}

const G4VSolid* X4GDMLReadStructure::ReadSolid(const char* path)
{
    X4GDMLReadStructure reader ; 
    reader.readFile(path);
    return reader.getSolid(-1); 
}

const G4VSolid* X4GDMLReadStructure::ReadSolidFromString(const char* gdmlstring)
{
    X4GDMLReadStructure reader ; 
    reader.readString(gdmlstring);
    return reader.getSolid(-1); 
}


void X4GDMLReadStructure::dumpMatrixMap(const char* msg) const 
{
    LOG(info) << msg ; 

    unsigned edgeitems = 5 ; 
    typedef std::map<G4String,G4GDMLMatrix>::const_iterator IT ; 
    for(IT it=matrixMap.begin() ; it != matrixMap.end() ; it++)
    {
        std::cout << it->first << std::endl ;
        const G4GDMLMatrix& matrix = it->second ; 

        X4GDMLMatrix xmatrix(matrix); 
        std::cout << xmatrix.desc(edgeitems) << std::endl ;  
    }
}

// the below is private (not protected) in G4GDMLReadSolids so cannot dump 
// std::map<G4String, G4MaterialPropertyVector*> mapOfMatPropVects;


