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


#include "G4TheRayTracer.hh"


#include "BFile.hh"
#include "OpticksHub.hh"

#include "CG4.hh"
#include "CRayTracer.hh"

#include "PLOG.hh"

CRayTracer::CRayTracer(CG4* g4)
    :
    m_g4(g4),
    m_ok(g4->getOpticks()),
    m_hub(g4->getHub()),
    m_composition(m_hub->getComposition()),

    m_figmaker(NULL),
    m_scanner(NULL),
    m_tracer( new G4TheRayTracer(m_figmaker, m_scanner) )
{
}


void CRayTracer::snap() const 
{
    std::string path_ = BFile::FormPath("$TMP/CRayTracer.jpeg"); 

    LOG(info) << "path " << path_ ; 
 
    G4String path = path_ ; 

    m_tracer->Trace( path );
  
}


