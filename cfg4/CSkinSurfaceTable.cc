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

#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>

#include "PLOG.hh"

#include "CSkinSurfaceTable.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4OpticalSurface.hh"

const plog::Severity CSkinSurfaceTable::LEVEL = PLOG::EnvLevel("CSkinSurfaceTable", "DEBUG") ; 


CSkinSurfaceTable::CSkinSurfaceTable()
    :
    CSurfaceTable("skin")
{
    init();
}
   
void CSkinSurfaceTable::init()
{
    int nsurf = G4LogicalSkinSurface::GetNumberOfSkinSurfaces();
    LOG(LEVEL) << " nsurf " << nsurf ; 
    const G4LogicalSkinSurfaceTable* sst = G4LogicalSkinSurface::GetSurfaceTable();

    assert( int(sst->size()) == nsurf );

    for(int i=0 ; i < nsurf ; i++)
    {
        G4LogicalSkinSurface* ss = (*sst)[i] ;
        const G4OpticalSurface* os = dynamic_cast<const G4OpticalSurface*>(ss->GetSurfaceProperty());
        add(os);

        const G4LogicalVolume* lv = ss->GetLogicalVolume() ;

        if( LEVEL > info )
        std::cout << std::setw(5) << i 
                  << std::setw(35) << ( ss ? ss->GetName() : "NULL" )
                  << std::setw(35) << ( os ? os->GetName() : "NULL" )
                  << " lv " << ( lv ? lv->GetName() : "NULL" ) 
                  << std::endl 
                  ;
    }
}

void CSkinSurfaceTable::dump(const char* msg)
{
    LOG(info) << msg << " numSurf " << getNumSurf() ; 
}

