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
#include "G4LogicalSkinSurface.hh"   

#include "X4.hh"
#include "X4LogicalSurface.hh"
#include "X4LogicalSkinSurface.hh"
#include "X4OpticalSurface.hh"

#include "GOpticalSurface.hh"   
#include "GSkinSurface.hh"   

#include "PLOG.hh"


GSkinSurface* X4LogicalSkinSurface::Convert(const G4LogicalSkinSurface* src, bool standardized )
{
    const char* name = X4::Name( src ); 
    size_t index = X4::GetOpticksIndex( src ) ;  

    G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
    assert( os );
    GOpticalSurface* optical_surface = X4OpticalSurface::Convert(os);   ; 
    assert( optical_surface );

    GSkinSurface* dst = new GSkinSurface( name, index, optical_surface) ;  
    // standard domain is set by GSkinSurface::init

    X4LogicalSurface::Convert( dst, src, standardized );

    const G4LogicalVolume* lv = src->GetLogicalVolume();

   
    /*
    LOG(fatal) 
         << " X4::Name(lv)  " << X4::Name(lv)
         << " X4::BaseNameAsis(lv) " << X4::BaseNameAsis(lv)
         ; 
    */

    dst->setSkinSurface(  X4::BaseNameAsis(lv) ) ; 


    return dst ; 
}

int X4LogicalSkinSurface::GetItemIndex( const G4LogicalSkinSurface* src )
{
    const G4LogicalSkinSurfaceTable* vec = G4LogicalSkinSurface::GetSurfaceTable() ; 
    return X4::GetItemIndex<G4LogicalSkinSurface>( vec, src ); 
}



