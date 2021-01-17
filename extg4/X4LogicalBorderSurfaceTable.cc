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


#include <string>
#include <iomanip>

#include "G4Version.hh"
#include "G4LogicalBorderSurface.hh"

#include "SGDML.hh"

#include "X4NameOrder.hh"
#include "X4LogicalBorderSurfaceTable.hh"
#include "X4LogicalBorderSurface.hh"

#include "GBorderSurface.hh"
#include "GSurfaceLib.hh"

#include "PLOG.hh"


const plog::Severity X4LogicalBorderSurfaceTable::LEVEL = PLOG::EnvLevel("X4LogicalBorderSurfaceTable","DEBUG"); 

void X4LogicalBorderSurfaceTable::Convert( GSurfaceLib* dst )
{
    X4LogicalBorderSurfaceTable xtab(dst); 
}

/**
X4LogicalBorderSurfaceTable::PrepareVector
-------------------------------------------

Prior to Geant4 1070 the G4LogicalBorderSurfaceTable type was a std::vector, 
for 1070 and above the table type changed to become a std::map with  
pair of pointers key : std::pair<const G4VPhysicalVolume*, const G4VPhysicalVolume*>.
As the std::map iteration order with such a key could potentially change from 
invokation to invokation or between platforms depending on where the pointer 
addresses got allocated it is necessary to impose a more meaningful 
and consistent order. 

As Opticks serializes all geometry objects into arrays for upload 
to GPU buffers and textures and uses indices to reference into these 
buffers and textures it is necessary for all collections of geometry objects 
to have well defined and consistent ordering.
To guarantee this the std::vector obtained from the std::map is sorted based on 
the 0x stripped name of the G4LogicalBorderSurface.

**/

const std::vector<G4LogicalBorderSurface*>* X4LogicalBorderSurfaceTable::PrepareVector(const G4LogicalBorderSurfaceTable* tab ) 
{
    typedef std::vector<G4LogicalBorderSurface*> VBS ; 
#if G4VERSION_NUMBER >= 1070
    typedef std::pair<const G4VPhysicalVolume*, const G4VPhysicalVolume*> PPV ; 
    typedef std::map<PPV, G4LogicalBorderSurface*>::const_iterator IT ; 

    VBS* vec = new VBS ;  
    for(IT it=tab->begin() ; it != tab->end() ; it++ )
    {
        G4LogicalBorderSurface* bs = it->second ;    
        vec->push_back(bs);        
        const PPV ppv = it->first ; 
        assert( ppv.first == bs->GetVolume1());  
        assert( ppv.second == bs->GetVolume2());  
    }

    {
        bool reverse = false ; 
        bool strip = true ; 
        X4NameOrder<G4LogicalBorderSurface> name_order(reverse, strip); 
        std::sort( vec->begin(), vec->end(), name_order ); 
        X4NameOrder<G4LogicalBorderSurface>::Dump("X4LogicalBorderSurfaceTable::PrepareVector after sort", *vec ); 
    }

#else
    const VBS* vec = tab ;  
    // hmm maybe should name sort pre 1070 too for consistency 
    // otherwise they will stay in creation order
    // Do this once 107* becomes more relevant to Opticks.
#endif
    return vec ; 
}

X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(GSurfaceLib* dst )
    :
    m_src(PrepareVector(G4LogicalBorderSurface::GetSurfaceTable())),
    m_dst(dst)
{
    init();
}

void X4LogicalBorderSurfaceTable::init()
{
    unsigned num_src = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ; 
    assert( num_src == m_src->size() );

    LOG(LEVEL) << " NumberOfBorderSurfaces " << num_src ;  
    
    for(size_t i=0 ; i < m_src->size() ; i++)
    {

        G4LogicalBorderSurface* src = (*m_src)[i] ; 

        LOG(LEVEL) << src->GetName() ; 

        GBorderSurface* dst = X4LogicalBorderSurface::Convert( src );

        assert( dst ); 

        m_dst->add(dst) ; // GSurfaceLib
    }
}







