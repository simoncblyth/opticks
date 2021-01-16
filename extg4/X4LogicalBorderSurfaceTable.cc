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


#include "G4Version.hh"
#include "G4LogicalBorderSurface.hh"

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

Prior to Geant4 1070 the G4LogicalBorderSurfaceTable type
was a std::vector, for 1070 and above the table changed to a std::map 
requiring conversion to a std::vector in order to have a well defined order.

The problem with the below conversion approach is that the ordering is 
potentially unreliable, changing from invokation to invokation 
and not matching between platforms. The std::map has an order determined 
by key comparisons, but when the key is a pair of pointers it is anyones 
guess what the order will be and how consistent it will be.

**/

const std::vector<G4LogicalBorderSurface*>* X4LogicalBorderSurfaceTable::PrepareVector(const G4LogicalBorderSurfaceTable* tab) 
{
#if G4VERSION_NUMBER >= 1070
    typedef std::vector<G4LogicalBorderSurface*> VBS ; 
    typedef std::pair<const G4VPhysicalVolume*, const G4VPhysicalVolume*> PPV ; 
    typedef std::map<PPV, G4LogicalBorderSurface*>  PPVBS ;   
    typedef std::map<PPV, G4LogicalBorderSurface*>::const_iterator IT ; 

    VBS* vec = new VBS ;  
    for(IT it=tab->begin() ; it != tab->end() ; it++ )
    {
        PVPV* pvpv = it.first ; 
        G4LogicalBorderSurface* bs = it.second ;    
        vec->push_back(bs);        
    }
    return vec ; 
#else
    return tab ; 
#endif
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







