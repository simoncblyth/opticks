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

// brap-
//#include "BTimeKeeper.hh"

// npy-
//#include "NGLM.hpp"
//#include "GLMFormat.hpp"
//#include "GLMPrint.hpp"
//#include "NSlice.hpp"

// okc-
#include "Opticks.hh"
//#include "Composition.hh"
//#include "OpticksConst.hh"
//#include "OpticksResource.hh"
//#include "OpticksAttrSeq.hh"
//#include "OpticksCfg.hh"

// okg-
#include "OpticksHub.hh"

// ggeo-
//#include "GGeoLib.hh"
//#include "GSurfaceLib.hh"

//#include "GMergedMesh.hh"
#include "GGeo.hh"

// opticksgeo-
#include "OpticksGeometry.hh"


#include "PLOG.hh"

const plog::Severity OpticksGeometry::LEVEL = PLOG::EnvLevel("OpticksGeometry", "DEBUG") ; 


OpticksGeometry::OpticksGeometry(OpticksHub* hub)
    :
    m_hub(hub),
    m_ok(m_hub->getOpticks()),
    m_composition(m_hub->getComposition()),
    m_ggeo(NULL),
    m_verbosity(m_ok->getVerbosity())
{
    init();
}

void OpticksGeometry::init()
{
    m_ggeo = new GGeo(m_ok);
    m_ggeo->setLookup(m_hub->getLookup());
}

GGeo* OpticksGeometry::getGGeo()
{
   return m_ggeo ; 
}

void OpticksGeometry::loadGeometry()
{

    LOG(LEVEL) << "["  ; 
    OK_PROFILE("_OpticksGeometry::loadGeometry");

    m_ggeo->loadGeometry();   // potentially from cache : for gltf > 0 loads both tri and ana geometry 

    if(!m_ggeo->isValid())
    {
        LOG(warning) << "invalid geometry, try creating geocache with --nogeocache/-G option " ; 
        m_ok->setExit(true); 
        return ; 
    }

    if(!m_ok->isGeocacheEnabled())
    {
        LOG(info) << "early exit due to --nogeocache/-G option " ; 
        m_ok->setExit(true); 
    }

    OK_PROFILE("OpticksGeometry::loadGeometry");
    LOG(LEVEL) << "]" ; 
}



