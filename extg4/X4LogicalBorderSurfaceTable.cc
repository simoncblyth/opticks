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

X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(GSurfaceLib* dst )
    :
    m_src(G4LogicalBorderSurface::GetSurfaceTable()),
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


