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

#include "CBorderSurfaceTable.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4OpticalSurface.hh"


const plog::Severity CBorderSurfaceTable::LEVEL = PLOG::EnvLevel("CBorderSurfaceTable", "DEBUG"); 

CBorderSurfaceTable::CBorderSurfaceTable()
    :
    CSurfaceTable("border")
{
    init();
}
   
void CBorderSurfaceTable::init()
{
    int nsurf = G4LogicalBorderSurface::GetNumberOfBorderSurfaces();

    LOG(LEVEL) << " nsurf " << nsurf ; 

    const G4LogicalBorderSurfaceTable* bst = G4LogicalBorderSurface::GetSurfaceTable();

    assert( int(bst->size()) == nsurf );

    for(int i=0 ; i < nsurf ; i++)
    {
        G4LogicalBorderSurface* bs = (*bst)[i] ;
        G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(bs->GetSurfaceProperty());
        add(os);

        if(LEVEL > info)
        {
            const G4VPhysicalVolume* pv1 = bs->GetVolume1() ;
            const G4VPhysicalVolume* pv2 = bs->GetVolume2() ;
            std::cout << std::setw(5) << i 
                      << std::setw(35) << bs->GetName()
                      << std::setw(35) << os->GetName()
                      ;

           if(pv1) std::cout << " pv1 " << pv1->GetName() << " #" << pv1->GetCopyNo()  ;
           else std::cout << " pv1 NULL " ;

           if(pv2) std::cout << " pv2 " << pv2->GetName() << " #" << pv2->GetCopyNo()  ;
           else std::cout << " pv2 NULL " ;

      
           std::cout << std::endl ;
       }
    }
}

void CBorderSurfaceTable::dump(const char* msg)
{
    LOG(info) << msg << " numSurf " << getNumSurf() ; 
}

