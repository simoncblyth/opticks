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


#include "G4LogicalSkinSurface.hh"

#include "X4LogicalSkinSurfaceTable.hh"
#include "X4LogicalSkinSurface.hh"

#include "GSkinSurface.hh"
#include "GSurfaceLib.hh"

#include "PLOG.hh"


void X4LogicalSkinSurfaceTable::Convert( GSurfaceLib* dst )
{
    X4LogicalSkinSurfaceTable x(dst); 
}

X4LogicalSkinSurfaceTable::X4LogicalSkinSurfaceTable(GSurfaceLib* dst )
    :
    m_src(G4LogicalSkinSurface::GetSurfaceTable()),
    m_dst(dst)
{
    init();
}


void X4LogicalSkinSurfaceTable::init()
{
    unsigned num_src = G4LogicalSkinSurface::GetNumberOfSkinSurfaces() ; 
    assert( num_src == m_src->size() );

    LOG(debug) << " NumberOfSkinSurfaces num_src " << num_src ;  
    
    for(size_t i=0 ; i < m_src->size() ; i++)
    {
        G4LogicalSkinSurface* src = (*m_src)[i] ; 

        //LOG(info) << src->GetName() ; 

        GSkinSurface* dst = X4LogicalSkinSurface::Convert( src );

        assert( dst ); 

        m_dst->add(dst) ; // GSurfaceLib
    }
}

