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



#include "OKConf_Config.hh"

#include "BTimeKeeper.hh"

#include "SSys.hh"
#include "SLog.hh"
#include "OXPPNS.hh"
#include "OError.hh"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksCfg.hh"

// okg-
#include "OpticksHub.hh"
#include "GScintillatorLib.hh"

// oxrap-
#include "OContext.hh"
#include "OFunc.hh"
#include "OColors.hh"
#include "OGeo.hh"
#include "OBndLib.hh"
#include "OScintillatorLib.hh"
#include "OSourceLib.hh"
#include "OBuf.hh"
#include "OConfig.hh"

#include "OScene.hh"


#include "PLOG.hh"


const plog::Severity OScene::LEVEL = debug ; 

OContext* OScene::getOContext()
{
    return m_ocontext ; 
}

OBndLib*  OScene::getOBndLib()
{
    return m_olib ; 
}

int OScene::preinit() const
{
    OKI_PROFILE("_OScene::OScene");  
    return 0 ; 
}  

OScene::OScene(OpticksHub* hub, const char* cmake_target, const char* ptxrel) 
    :   
    m_preinit(preinit()),
    m_log(new SLog("OScene::OScene","", LEVEL)),
    m_timer(new BTimeKeeper("OScene::")),
    m_hub(hub),
    m_ok(hub->getOpticks()),
    m_ocontext(OContext::Create(m_ok, cmake_target, ptxrel)),
    m_osolve(NULL),
    m_ocolors(NULL),
    m_ogeo(NULL),
    m_olib(NULL),
    m_oscin(NULL),
    m_osrc(NULL),
    m_verbosity(m_ok->getVerbosity()),
    m_use_osolve(false)
{
    init();
    (*m_log)("DONE");
}


/**

OScene::Init
---------------

1. creates OptiX context
2. instanciates the O*Libs which populate the OptiX context 
   from the corresponding libs provided by OpticksHub accessors
   (NB not directly from GGeo or GScene, the Hub mediates)
::

    OColors 
    OSourceLib
    OScintillatorLib
    OGeo
    OBndLib 

**/



void OScene::init()
{
    LOG(info) << "[" ; 

    plog::Severity level = LEVEL ; 

    m_timer->setVerbose(true);
    m_timer->start();

    optix::Context context = m_ocontext->getContext(); 
    

    // solvers despite being used for geometry intersects have no dependencies
    // as just pure functions : so place them accordingly 
    if(m_use_osolve)
    {  
        m_osolve = new OFunc(m_ocontext, "solve_callable.cu", "solve_callable", "SolveCubicCallable" ) ; 
        m_osolve->convert();
    }

    LOG(LEVEL) 
          << " ggeobase identifier : " << m_hub->getIdentifier()
          ;

    LOG(level) << "(OColors)" ;
    m_ocolors = new OColors(context, m_ok->getColors() );
    m_ocolors->convert();

    // formerly did OBndLib here, too soon

    LOG(level) << "(OSourceLib)" ;
    m_osrc = new OSourceLib(context, m_hub->getSourceLib());
    m_osrc->convert();


    GScintillatorLib* sclib = m_hub->getScintillatorLib() ;
    unsigned num_scin = sclib->getNumScintillators(); 
    const char* slice = "0:1" ;

    LOG(level) << "(OScintillatorLib)"
               << " num_scin " << num_scin 
               << " slice " << slice  
               ;

    // a placeholder reemission texture is created even when no scintillators
    m_oscin = new OScintillatorLib(context, sclib );
    m_oscin->convert(slice);


    LOG(level) << "(OGeo)" ;
    m_ogeo = new OGeo(m_ocontext, m_ok, m_hub->getGeoLib() );
    LOG(level) << "(OGeo) convert" ;
    m_ogeo->convert();
    LOG(level) << "(OGeo) done" ;


    LOG(level) << "(OBndLib)" ;
    m_olib = new OBndLib(context,m_hub->getBndLib());
    m_olib->convert();
    // this creates the BndLib dynamic buffers, which needs to be after OGeo
    // as that may add boundaries when using analytic geometry


    LOG(debug) << m_ogeo->description();

    LOG(info) << "]" ;

    OKI_PROFILE("OScene::OScene");  
}


void OScene::cleanup()
{
    delete m_ocontext ;
}


