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
#include "G4LogicalSurface.hh"
#include "G4MaterialPropertiesTable.hh"

#include "X4LogicalSurface.hh"

#include "X4MaterialPropertiesTable.hh"
#include "GPropertyMap.hh"
#include "PLOG.hh"

const plog::Severity X4LogicalSurface::LEVEL = PLOG::EnvLevel("X4LogicalSurface","DEBUG") ; 

void X4LogicalSurface::Convert(GPropertyMap<float>* dst,  const G4LogicalSurface* src)
{
    LOG(LEVEL) << "[" ; 
    const G4SurfaceProperty*  psurf = src->GetSurfaceProperty() ;   
    const G4OpticalSurface* opsurf = dynamic_cast<const G4OpticalSurface*>(psurf);
    assert( opsurf );   
    G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable() ;
    X4MaterialPropertiesTable::Convert( dst, mpt );

    LOG(LEVEL) << "]" ; 
}



