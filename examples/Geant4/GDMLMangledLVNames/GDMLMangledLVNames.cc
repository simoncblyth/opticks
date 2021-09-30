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

/**
GDMLMangledLVNames.cc
=======================

This failed to reproduce a problem of mangled LV names 
seen in the switch to 10.4.2 see: 

* notes/issues/Geant4_update_to_10_4_2.rst

But it did reveal interference between some PLOG.hh dangerous 
define of trace and xercesc headers, see:

* notes/issues/PLOG_dangerous_trace_define.rst

And fixing that issue fixed the mangled names problem, so 
this was a useful check even though it didnt reproduce the problem

**/


#include "DetectorConstruction.hh"

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"

#include "BFile.hh"

#include "OPTICKS_LOG.hh"
#include "G4GDMLParser.hh"

void write_gdml( const G4VPhysicalVolume* pv, const char* path )
{
    LOG(info) << "export to " << path ; 

    G4GDMLParser gdml ;
    G4bool refs = true ;
    G4String schemaLocation = "" ; 

    gdml.Write(path, pv, refs, schemaLocation );
}

G4VPhysicalVolume* read_gdml( const char* path )
{
    G4GDMLParser gdml ;
    bool validate = false ; 
    bool trimPtr = false ; 
    gdml.SetStripFlag(trimPtr);
    gdml.Read(path, validate);
    return gdml.GetWorldVolume() ;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    DetectorConstruction dc ;  

    G4VPhysicalVolume* pv = dc.Construct() ; 
    G4LogicalVolume* lv = pv->GetLogicalVolume(); 

    LOG(info) << " pv " << pv->GetName() ; 
    LOG(info) << " lv " << lv->GetName() ; 

    const char* path = "$TMP/examples/Geant4/GDMLMangledLVNames.gdml" ; 
    BFile::RemoveFile(path); 
    std::string apath = BFile::preparePath(path);

    write_gdml( pv, apath.c_str() ); 


    G4VPhysicalVolume* pv2 = read_gdml( apath.c_str() ) ;
    G4LogicalVolume* lv2 = pv2->GetLogicalVolume(); 

    LOG(info) << " pv2 " << pv2->GetName() ; 
    LOG(info) << " lv2 " << lv2->GetName() ; 

    return 0 ; 
}


