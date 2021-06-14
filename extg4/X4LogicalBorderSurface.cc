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

#include <cassert>

#include "G4OpticalSurface.hh"   
#include "G4LogicalBorderSurface.hh"   

#include "X4.hh"
#include "X4LogicalSurface.hh"
#include "X4LogicalBorderSurface.hh"
#include "X4LogicalBorderSurfaceTable.hh"
#include "X4OpticalSurface.hh"

#include "GOpticalSurface.hh"   
#include "GBorderSurface.hh"   
#include "GDomain.hh"   

#include "PLOG.hh"


const plog::Severity X4LogicalBorderSurface::LEVEL = PLOG::EnvLevel("X4LogicalBorderSurface", "DEBUG"); 


GBorderSurface* X4LogicalBorderSurface::Convert(const G4LogicalBorderSurface* src)
{
    const char* name = X4::Name( src ); 
    size_t index = X4::GetOpticksIndex( src ) ;  

    G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
    assert( os );
    GOpticalSurface* optical_surface = X4OpticalSurface::Convert(os);   ; 
    assert( optical_surface );

    GBorderSurface* dst = new GBorderSurface( name, index, optical_surface) ;  
    // standard domain is set by GBorderSurface::init

    X4LogicalSurface::Convert( dst, src);

    const G4VPhysicalVolume* pv1 = src->GetVolume1(); 
    const G4VPhysicalVolume* pv2 = src->GetVolume2(); 
    assert( pv1 && pv2 ) ; 

    dst->setBorderSurface( X4::Name(pv1), X4::Name(pv2) );   

    LOG(LEVEL) << name << " is_sensor " << dst->isSensor() ; 

    return dst ; 
}


/**
X4LogicalBorderSurface::GetItemIndex
-------------------------------------

Returns index of the border surface within the Geant4 vector of all border surfaces
obtained from the border surface table.
As a border surface only index this does not seem very useful.  
Perhaps thats why it appears not to be used in anger.

**/

int X4LogicalBorderSurface::GetItemIndex( const G4LogicalBorderSurface* src )  // static 
{
    const G4LogicalBorderSurfaceTable* tab = G4LogicalBorderSurface::GetSurfaceTable() ; 

    typedef std::vector<G4LogicalBorderSurface*> VBS ; 

    const VBS* vec = X4LogicalBorderSurfaceTable::PrepareVector(tab) ; 

    return X4::GetItemIndex<G4LogicalBorderSurface>( vec, src ); 
}




