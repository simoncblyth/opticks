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

#include "OPTICKS_LOG.hh"
#include "BOpticksResource.hh"
#include "CameraCfg.hh"
#include "Camera.hh"
#include "View.hh"

#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "Composition.hh"
#include "BCfg.hh"


void test_configure_one(int argc, char** argv)
{
    Camera camera(1920,1080) ; 
    camera.Print("initial default camera");

    bool live = true ;  
    CameraCfg<Camera>* cfg = new CameraCfg<Camera>("camera", &camera, live);
    cfg->setVerbose(true); 

    const char* path = "$PREFIX/include/OpticksCore/CfgTest.cfg" ; 
    LOG(error) << " configfile : " << path ; 
    cfg->configfile(path);

    camera.Print("after configfile");

    cfg->commandline(argc,argv);
    camera.Print("after commandline");

    cfg->liveline("--check-non-existing 123 --near 777 --far 7777 --zoom 7 --type 1");
    camera.Print("after liveline");
}


BCfg* test_configure_tree(Composition* m_composition)
{
    Opticks*  m_ok = m_composition->getOpticks(); 

    // the below instanciations canonically in OpticksHub::OpticksHub  
    BCfg*                 m_cfg = new BCfg("umbrella", false);  
    OpticksCfg<Opticks>*  m_fcfg = m_ok->getCfg() ; // elephant with hundreds of options 
      
    // canoncially OpticksHub::init/OpticksHub::configure
    m_cfg->add(m_fcfg);               // put elephant under the umbrella 

    m_composition->addConfig(m_cfg);  // m_cfg collects the BCfg subclass objects such as ViewCfg,CameraCfg etc.. from Composition

    int argc    = m_ok->getArgc();
    char** argv = m_ok->getArgv();

    m_cfg->commandline(argc, argv);

    m_ok->configure();        // <--  HUH ? its already added to the umbrella

    if(m_fcfg->hasError())
    {   
        LOG(fatal) << "parse error " << m_fcfg->getErrorMessage() ; 
        m_fcfg->dump("OpticksHub::config m_fcfg");
        m_ok->setExit(true);
    }

    return m_cfg ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    // BOpticksResource rsc ;   // needed for BFile::ResolveKey when not using Opticks class
    //test_configure_one(argc, argv); 

    Composition*   m_composition = new Composition(&ok); 
    BCfg*          m_umbrella = test_configure_tree(m_composition); 

    View* m_view = m_composition->getView(); 
    m_view->Summary(); 

    char* liveline = getenv("LIVELINE") ; 
    if(liveline)
    {
        m_umbrella->liveline(liveline);  
        m_view->Summary("after liveline"); 
    }

    return 0 ; 
}

// om-;TEST=CfgTest om-t 
// om-;TEST=CfgTest BCfg=INFO View=INFO om-t 
// LIVELINE="--eye 0.3,0.3,0.3 --look 0.5,0,0" CfgTest --eye 1,0,0 --look 0,0,1 --up 1,1,1 



