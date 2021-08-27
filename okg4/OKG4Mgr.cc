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

#include "OKG4Mgr.hh"

class NConfigurable ; 

#include "SLog.hh"
#include "BTimeKeeper.hh"

#include "G4StepNPY.hpp"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"
#include "OpticksRun.hh"    

#include "OpticksHub.hh"    // okg-
#include "OpticksIdx.hh"    
#include "OpticksGen.hh"    

#include "OKPropagator.hh"  // ok-

#define GUI_ 1
#include "OpticksViz.hh"

#include "CG4.hh"
#include "CGenstep.hh"
#include "CGenerator.hh"

#include "PLOG.hh"
#include "OKG4_BODY.hh"

const plog::Severity OKG4Mgr::LEVEL = PLOG::EnvLevel("OKG4Mgr", "DEBUG"); 


int OKG4Mgr::rc() const 
{
    return m_ok->rc();
}

/**
OKG4Mgr::OKG4Mgr
------------------

m_hub(OpticksHub)
    loads geometry from geocache into GGeo 

m_g4(CG4)
    when "--load" option is NOT used (TODO:change "--load" to "--loadevent" ) 
    geometry is loaded from GDML into Geant4 model by CGeometry/CGDMLDetector
    The .gdml file was persisted into geocache at its creation, from
    G4Opticks::translateGeometry with CGDML::Export prior to populating GGeo

m_viz(OpticksViz)
    when "--compute" option is NOT used instanciate from m_hub    


Note frailty of having two sources of geometry here. I recall previous
bi-simulation matching activity where I avoided this by creating the Geant4 geometry 
from the Opticks one : but I think that was just for simple test geometries. 

Of course the geocache was created from the same initial source Geant4 geometry,
but still this means more layers of code, but its inevitable for bi-simulation
which is the point of OKG4Mgr. 

*Perhaps a more direct way...*

Hmm do I need OKX4Mgr ?  To encapsulate whats done in OKX4Test and make it reusable.
That starts from GDML uses G4GDMLParser to get G4VPhysicalVolume 
does the direct X4 conversion to populate a GGeo, persists to cache and 
then uses OKMgr to pop the geometry up to GPU for propagation.
 
**/

int OKG4Mgr::preinit() const
{
    OK_PROFILE("_OKG4Mgr::OKG4Mgr"); 
    return 0 ;  
}

OKG4Mgr::OKG4Mgr(int argc, char** argv) 
    :
    m_log(new SLog("OKG4Mgr::OKG4Mgr","",debug)),
    m_ok(new Opticks(argc, argv)),  
    m_preinit(preinit()),
    m_run(m_ok->getRun()),
    m_hub(new OpticksHub(m_ok)),            // configure, loadGeometry and setupInputGensteps immediately
    m_load(m_ok->isLoad()),
    m_nog4propagate(m_ok->isNoG4Propagate()),
    m_nogpu(m_ok->isNoGPU()),
    m_production(m_ok->isProduction()), 
    m_idx(new OpticksIdx(m_hub)),
    m_num_event(m_ok->getMultiEvent()),     // huh : m_gen should be in change of the number of events ? 
    m_gen(m_hub->getGen()),
    m_g4(        m_load ? NULL : new CG4(m_hub)),   // configure and initialize immediately 
    m_generator( m_load ? NULL : m_g4->getGenerator()),
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),    // true: load/create Bookmarks, setup shaders, upload geometry immediately 
    m_propagator(m_nogpu ? nullptr : new OKPropagator(m_hub, m_idx, m_viz))
{
    (*m_log)("DONE");
    init(); 
}

void OKG4Mgr::init() const
{
    OK_PROFILE("OKG4Mgr::OKG4Mgr"); 
}





OKG4Mgr::~OKG4Mgr()
{
    cleanup();
}


void OKG4Mgr::propagate()
{
    if(m_load)
    {   
         char load_ctrl = m_ok->hasOpt("vizg4|evtg4") ? '-' : '+' ;

         m_run->loadEvent(load_ctrl); 

         if(m_viz) 
         {   
             m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt

             m_viz->uploadEvent(load_ctrl);      // not needed when propagating as event is created directly on GPU

             m_viz->indexPresentationPrep();
         }   
    }   
    else if(m_ok->isNoPropagate())
    {   
        LOG(info) << "--nopropagate/-P" ;
    }   
    else if(m_num_event > 0)
    {
        char ctrl = '=' ;  

        for(int i=0 ; i < m_num_event ; i++) 
        {   
            propagate_();

            if(m_ok->isSave())
            {
                m_ok->saveEvent(ctrl);
                if(!m_production)  m_hub->anaEvent();
            }
            m_ok->resetEvent(ctrl);

        }
        m_ok->postpropagate();
    }   
}




/**
OKG4Mgr::propagate_
---------------------

THIS NEEDS RETHINK AS DOES NOT 
FIT WITH THE CManager APPROACH 


Hmm propagate implies just photons to me, so this name 
is misleading as it does a G4 beamOn with the hooked up 
CSource subclass providing the primaries, which can be 
photons but not necessarily. 

Normally the G4 propagation is done first, because 
gensteps eg from G4Gun can then be passed to Opticks.
However with RNG-aligned testing using "--align" option
which uses emitconfig CPU generated photons there is 
no need to do G4 first. Actually it is more convenient
for Opticks to go first in order to allow access to the ucf.py 
parsed  kernel pindex log during lldb python scripted G4 debugging.
 
Hmm it would be cleaner if m_gen was in charge if the branching 
here as its kinda similar to initSourceCode.

Notice the different genstep handling between this and OKMgr 
because this has G4 available, so gensteps can come from the
horses mouth.

TOFIX: when using --nog4propagate it should not be necessary to
instanciate m_g4 and m_generator : but the machinery forces to do so


**/

void OKG4Mgr::old_style_propagate_()
{ 
    bool align = m_ok->isAlign();

    if(m_generator->hasGensteps())   // TORCH
    {
         NPY<float>* gs = m_generator->getGensteps() ;
         m_ok->createEvent(gs, '=');

         unsigned numPhotons = G4StepNPY::CountPhotons(gs); 
         LOG(LEVEL) 
             << " G4StepNPY::CountPhotons " << numPhotons
             ;   

         CGenstep cgs = m_g4->addGenstep( numPhotons, 'T' ); 

         LOG(info)
             << " numPhotons " << numPhotons 
             << " cgs " << cgs.desc()
             ;           

         if(align && m_propagator)
             m_propagator->propagate();

         if(!m_nog4propagate) 
             m_g4->propagate();
    }
    else   // no-gensteps : G4GUN or PRIMARYSOURCE
    {
         NPY<float>* gs = m_g4->propagate() ; ;

         if(!gs) LOG(fatal) << "CG4::propagate failed to return gensteps" ; 
         assert(gs);

         m_ok->createEvent(gs, '=' );
    }
            
    if(!align && m_propagator)
        m_propagator->propagate();
}


void OKG4Mgr::propagate_()
{
    LOG(LEVEL) << "[" ; 
    unsigned numPhotons = 100 ; 
    int node_index = 0 ; 
    unsigned originTrackID = 101 ; 
    CGenstep cgs = m_g4->collectDefaultTorchStep(numPhotons, node_index, originTrackID );  

    LOG(LEVEL)
        << " numPhotons " << numPhotons 
        << " cgs " << cgs.desc()
        ;           

    if(!m_nog4propagate) 
        m_g4->propagate();

    if(m_propagator)
        m_propagator->propagate();

    LOG(LEVEL) << "]" ; 
}



void OKG4Mgr::visualize()
{
    if(m_viz) m_viz->visualize();
}

void OKG4Mgr::cleanup()
{
    m_propagator->cleanup();
    m_hub->cleanup();
    if(m_viz) m_viz->cleanup();
    m_ok->cleanup(); 
    if(m_g4) m_g4->cleanup(); 
}


