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
#include <cassert>

#include "CFG4_BODY.hh"
#include "CPrimaryCollector.hh"
#include "CSource.hh"
#include "Opticks.hh"

CSource::CSource(Opticks* ok )  
    :
    m_ok(ok),
    m_recorder(NULL),
    m_vtx_count(0)
{
}

CSource::~CSource()
{
}  

void CSource::setRecorder(CRecorder* recorder)
{
    m_recorder = recorder ;  
}


NPY<float>* CSource::getSourcePhotons() const
{
    return NULL ; 
}

void CSource::collectPrimaryVertex(const G4PrimaryVertex* vtx)
{
    //if( m_vtx_count % 1000 == 0 ) OK_PROFILE("CSource::collectPrimaryVertex_1k"); 

    CPrimaryCollector* pc = CPrimaryCollector::Instance() ;
    assert( pc ); 
    G4int vertex_index = 0 ;    // assumption 
    pc->collectPrimaryVertex(vtx, vertex_index); 

    m_vtx_count += 1 ; 
}



