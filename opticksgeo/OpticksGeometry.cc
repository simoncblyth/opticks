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
#include "BTimeKeeper.hh"

// npy-
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "NSlice.hpp"

// okc-
#include "Opticks.hh"
#include "Composition.hh"
#include "OpticksConst.hh"
#include "OpticksResource.hh"
#include "OpticksAttrSeq.hh"
#include "OpticksCfg.hh"

// okg-
#include "OpticksHub.hh"

// ggeo-

#include "GGeoLib.hh"
#include "GSurfaceLib.hh"

#include "GMergedMesh.hh"
#include "GGeo.hh"

// assimpwrap
#include "AssimpGGeo.hh"

// openmeshrap-
#include "MFixer.hh"
#include "MTool.hh"


// opticksgeo-
#include "OpticksGeometry.hh"


#include "PLOG.hh"

const plog::Severity OpticksGeometry::LEVEL = PLOG::EnvLevel("OpticksGeometry", "DEBUG") ; 





OpticksGeometry::OpticksGeometry(OpticksHub* hub)
   :
   m_hub(hub),
   m_ok(m_hub->getOpticks()),
   m_composition(m_hub->getComposition()),
   m_fcfg(m_ok->getCfg()),
   m_ggeo(NULL),

   m_verbosity(m_ok->getVerbosity())
{
    init();
}

GGeo* OpticksGeometry::getGGeo()
{
   return m_ggeo ; 
}


void OpticksGeometry::init()
{
    m_ggeo = new GGeo(m_ok);
    m_ggeo->setLookup(m_hub->getLookup());
}


void OpticksGeometry::loadGeometry()
{

    LOG(LEVEL) << "["  ; 
    OK_PROFILE("_OpticksGeometry::loadGeometry");

    loadGeometryBase(); //  usually from cache

    if(!m_ggeo->isValid())
    {
        LOG(warning) << "invalid geometry, try creating geocache with --nogeocache/-G option " ; 
        m_ok->setExit(true); 
        return ; 
    }


    // modifyGeometry moved up to OpticksHub

    fixGeometry();

    //registerGeometry moved up to OpticksHub

    if(!m_ok->isGeocacheEnabled())
    {
        LOG(info) << "early exit due to --nogeocache/-G option " ; 
        m_ok->setExit(true); 
    }


    OK_PROFILE("OpticksGeometry::loadGeometry");
    LOG(LEVEL) << "]" ; 
}


/**
OpticksGeometry::loadGeometryBase
------------------------------------



**/

void OpticksGeometry::loadGeometryBase()
{
    LOG(LEVEL) << "[" ; 
    OpticksResource* resource = m_ok->getResource();

    if(m_ok->hasOpt("qe1"))
        m_ggeo->getSurfaceLib()->setFakeEfficiency(1.0);


    m_ggeo->setLoaderImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr

    m_ggeo->setMeshJoinImp(&MTool::joinSplitUnion);
    m_ggeo->setMeshVerbosity(m_ok->getMeshVerbosity());    
    m_ggeo->setMeshJoinCfg( resource->getMeshfix() );

    std::string meshversion = m_fcfg->getMeshVersion() ;;
    if(!meshversion.empty())
    {
        LOG(error) << "using debug meshversion " << meshversion ;  
        m_ggeo->getGeoLib()->setMeshVersion(meshversion.c_str());
    }

    m_ggeo->loadGeometry();   // potentially from cache : for gltf > 0 loads both tri and ana geometry 
        
    if(m_ggeo->getMeshVerbosity() > 2)
    {
        GMergedMesh* mesh1 = m_ggeo->getMergedMesh(1);
        if(mesh1)
        {
            mesh1->dumpVolumes("OpticksGeometry::loadGeometryBase mesh1");
            mesh1->save("$TMP", "GMergedMesh", "baseGeometry") ;
        }
    }

    LOG(LEVEL) << "]" ; 
}


void OpticksGeometry::fixGeometry()
{
    if(m_ggeo->isLoadedFromCache())
    {
        LOG(debug) << "needs to be done precache " ;
        return ; 
    }
    LOG(info) << "[" ; 

    MFixer* fixer = new MFixer(m_ggeo);
    fixer->setVerbose(m_ok->hasOpt("meshfixdbg"));
    fixer->fixMesh();
 
    bool zexplode = m_ok->hasOpt("zexplode");
    if(zexplode)
    {
       // for --jdyb --idyb --kdyb testing : making the cleave OR the mend obvious
        glm::vec4 zexplodeconfig = gvec4(m_fcfg->getZExplodeConfig());
        print(zexplodeconfig, "zexplodeconfig");

        GMergedMesh* mesh0 = m_ggeo->getMergedMesh(0);
        mesh0->explodeZVertices(zexplodeconfig.y, zexplodeconfig.x ); 
    }

    LOG(info) << "]" ; 
}


