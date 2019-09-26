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


#include "X4SolidStore.hh"
#include "X4Solid.hh"
#include "PLOG.hh"
#include "Opticks.hh"
#include "G4SolidStore.hh"
#include "NNode.hpp"


void X4SolidStore::Dump() // static
{
    G4SolidStore* store = G4SolidStore::GetInstance() ; 
    assert( store ); 
    if(!store) return ; 

    unsigned num_solid = store->size(); 
    LOG(info) << " num_solid " << num_solid ; 

    Opticks* ok = Opticks::GetInstance(); 

    for(unsigned i=0 ; i < num_solid ; i++)
    {
        G4VSolid* solid = (*store)[i] ; 
        //LOG(info) << solid ; 

        nnode* tree = X4Solid::Convert(solid, ok); 
        LOG(info)  << tree->ana_brief() ;  

        if( i == num_solid - 2 )
        LOG(info)  << tree->ana_desc() ;  
    }
}


G4VSolid* X4SolidStore::Get(int index) // static
{
    G4SolidStore* store = G4SolidStore::GetInstance() ; 
    unsigned size = store->size()  ; 
    if( index < 0 ) index += size ; 
    unsigned idx = unsigned(index);  
    return idx < size ? (*store)[idx] : NULL ; 
} 
