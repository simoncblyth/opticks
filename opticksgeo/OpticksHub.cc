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

// sysrap-
#include "SLog.hh"
#include "SCtrl.hh"

// brap-
#include "BTimeKeeper.hh"
#include "BCfg.hh"
#include "BStr.hh"
#include "BMap.hh"

#include "NState.hpp"
#include "NLookup.hpp"
#include "NPY.hpp"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NGPU.hpp"

#define GLMVEC4(g) glm::vec4((g).x,(g).y,(g).z,(g).w) 


#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

// npy-
#include "TorchStepNPY.hpp"
#include "G4StepNPY.hpp"
#include "Index.hpp"


// numpyserver-
#ifdef OPTICKS_NPYSERVER
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"
#endif

// ggeo-
#include "GItemIndex.hh"
#include "GMergedMesh.hh"
#include "GGeoLib.hh"
#include "GNodeLib.hh"
#include "GGeo.hh"
#include "GGeoTest.hh"

// okc-
#include "Bookmarks.hh"
#include "FlightPath.hh"
#include "OpticksPhoton.h"
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksConst.hh"
#include "OpticksEvent.hh"
#include "OpticksColors.hh"
#include "OpticksActionControl.hh"
#include "Composition.hh"

// opticksgeo-
#include "OpticksHub.hh"
#include "OpticksGen.hh"
#include "OpticksRun.hh"
#include "OpticksAim.hh"
//#include "OpticksGeometry.hh"

#include "PLOG.hh"

const plog::Severity OpticksHub::LEVEL = debug ; 

/**

Formerly used GGeoBase as a common interface to GGeo/GScene/GGeoTest.
But since analytic/triangulated unification and the adoption of 
the direct workflow there is no more need for GScene and 
less need for GGeoBase. 

Although can regard GGeoBase as a formal "hyper public" kind of 
promotion of parts of the GGeo API that are not expected to change.

Although still have GGeo and GGeoTest ?

**/

const char* OpticksHub::getIdentifier()
{
    GGeoBase* ggb = getGGeoBase(); 
    return ggb->getIdentifier();
}
GMergedMesh* OpticksHub::getMergedMesh( unsigned index )
{
    GGeoBase* ggb = getGGeoBase(); 
    return ggb->getMergedMesh(index);
}

GNodeLib* OpticksHub::getNodeLib()
{
    GGeoBase* ggb = getGGeoBase();  
    return ggb->getNodeLib();
}
GMaterialLib* OpticksHub::getMaterialLib()
{  
    GGeoBase* ggb = getGGeoBase();
    return ggb->getMaterialLib() ; 
}
GSurfaceLib* OpticksHub::getSurfaceLib() 
{   
    GGeoBase* ggb = getGGeoBase();
    return ggb->getSurfaceLib() ; 
}

GBndLib* OpticksHub::getBndLib() 
{   
    GGeoBase* ggb = getGGeoBase();
    return ggb->getBndLib() ; 
}
GScintillatorLib* OpticksHub::getScintillatorLib() 
{ 
    GGeoBase* ggb = getGGeoBase();
    return ggb->getScintillatorLib() ;
}
GSourceLib* OpticksHub::getSourceLib() 
{ 
    GGeoBase* ggb = getGGeoBase();
    return ggb->getSourceLib() ;
}
GGeoLib* OpticksHub::getGeoLib()
{
    GGeoBase* ggb = getGGeoBase();
    return ggb->getGeoLib() ; 
}


void OpticksHub::setErr(int err)
{
    m_err = err ; 
}
int OpticksHub::getErr() const 
{
    return m_err ; 
}



/**
OpticksHub::command
-------------------

Invoked from lower levels, eg okc.InterpolatedView, on view switching via SCtrl protocol.
(OpticksHub ISA SCtrl, which is set as okc.InterpolatedView.m_ctrl allowing 
InterpolatedView::nextPair to send commands on high : up to here)
 
It would be better for this to live down in Composition, but it will take a while to 
get the requisite state all down there, so leaving up here for now.

Hmm but this is not high enough... 

For commandContentStyle need to prod the Scene, for the change to be acted upon 
Perhaps should move most of the below command handling down to Composition and 
move the frontdoor up to OpticksViz, so could do then easily prod the Scene ?


Hmm NConfigurable is doing something very similar to SCtrl and is already 
inplace for many classes.  TODO: combine these 

**/

void OpticksHub::command(const char* cmd) 
{
    assert( strlen(cmd) == 2 ); 
    m_composition->command(cmd); 
}


OpticksHub::OpticksHub(Opticks* ok) 
    :
    SCtrl(),
    m_log(new SLog("OpticksHub::OpticksHub","", LEVEL)),
    m_ok(ok),
#ifdef WITH_M_GLTF
    m_gltf(-1),        // m_ok not yet configured, so defer getting the settings
#endif
    m_run(m_ok->getRun()),
    m_ggeo(GGeo::GetInstance()),   // a pre-existing instance will prevent subsequent loading from cache   
    m_composition(new Composition(m_ok)),
#ifdef OPTICKS_NPYSERVER
    m_delegate(NULL),
    m_server(NULL)
#endif
    m_umbrella_cfg(new BCfg("umbrella", false)),
    m_fcfg(m_ok->getCfg()),
    m_state(NULL),
    m_lookup(new NLookup()),
    m_bookmarks(NULL),
    m_flightpath(NULL),
    m_gen(NULL),
    m_aim(NULL),
    m_geotest(NULL),
    m_err(0),
    m_ctrl(this)
{
    init();
    (*m_log)("DONE");
}


void OpticksHub::setCtrl(SCtrl* ctrl)
{
    m_ctrl = ctrl ; 
}

void OpticksHub::init()
{
    OK_PROFILE("_OpticksHub::init");  

    pLOG(LEVEL,0) << "[" ;   // -1 : one notch more easily seen than LEVEL

    add(m_fcfg);

    configure();
    configureServer();
    configureCompositionSize();


    if(m_ok->isLegacy()) 
    { 
        LOG(fatal) << m_ok->getLegacyDesc(); 
        configureLookupA();
    }

    m_aim = new OpticksAim(this) ; 

    if( m_ggeo == NULL )
    {
        loadGeometry() ;    
    }
    else
    {
        adoptGeometry() ;    
    }
    assert( m_ggeo ) ; 

    if(m_err) return ; 


    m_gen = new OpticksGen(this) ;

    pLOG(LEVEL,0) << "]" ; 
    OK_PROFILE("OpticksHub::init");  
}


/**
OpticksHub::loadGeometry
-------------------------


Formerly the OpticksGeometry intermediary was used here.

**/


void OpticksHub::loadGeometry()
{
    assert(m_ggeo == NULL && "OpticksHub::loadGeometry should only be called once");

    LOG(info) << "[ " << m_ok->getIdPath()  ; 

    m_ggeo = new GGeo(m_ok) ; 
    m_ggeo->setLookup(getLookup());  // TODO: see if legacy lookup stuff can be removed
    m_ggeo->loadGeometry();  

    bool valid = m_ggeo->isValid() ; 
    if(!valid) LOG(fatal) << "invalid geometry, try creating geocache with geocache-create and set OPTICKS_KEY " ; 
    assert(valid); 

    //   Lookup A and B are now set ...
    //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
    //      B : on GGeo loading in GGeo::setupLookup
    //

    if(m_ok->isTest())  // --test : instanciate GGeoTest 
    {
        setupTestGeometry(); 
    }
    else
    {
        LOG(LEVEL) << "NOT modifying geometry" ; 
    }

    m_ggeo->close();  // mlib and slib  (June 2018, following remove the auto-trigger-close on getIndex in the proplib )

    m_aim->registerGeometry( m_ggeo );
    
    m_ggeo->setComposition(m_composition);  
    // hmm: not keen on any changes to GGeo after loading, would prefer m_ggeo to be const 
    // perhaps composition should live in m_ok

    LOG(info) << "]" ; 
}


/**
OpticksHub::adoptGeometry
--------------------------

Adopts a directly created geometry, ie one that was not loaded from cache.
Now with G4Opticks this also adopts an externally loaded from cache GGeo.


**/


void OpticksHub::adoptGeometry()
{
    LOG(LEVEL) << "[" ; 

    assert( m_ggeo ); 

    assert( ( m_ggeo->isLoadedFromCache() || m_ggeo->isPrepared() ) && "MUST GGeo::prepare() live  geometry before adoption and subsequent GPU conversion " ) ;

    m_aim->registerGeometry( m_ggeo );
    
    m_ggeo->setComposition(m_composition);

    LOG(LEVEL) << "]" ; 
}


/**
OpticksHub::setupTestGeometry
-------------------------------

TODO: find a way to handle test geometry without such different code paths ?
possibly by making GGeoTest fulfil a common interface to GGeo and internally 
get most of its details from the basis GGeo. 

**/

void OpticksHub::setupTestGeometry()
{
    LOG(info) << "--test modifying geometry" ; 

    assert(m_geotest == NULL);

    GGeoBase* basis = getGGeoBasePrimary();  // downcast m_ggeo

    m_geotest = new GGeoTest(m_ok, basis);

    int err = m_geotest->getErr() ;
    if(err) 
    {
        setErr(err);
    }
}




std::string OpticksHub::desc() const 
{
    std::stringstream ss ; 

    GGeoBase* ggb = getGGeoBase(); 

    ss << "OpticksHub"
       << " encumbent " << ( ggb ? ggb->getIdentifier() : "-" ) 
       << " m_ggeo " << m_ggeo
       << " m_gen " << m_gen
       ;  

    return ss.str();
}


/**
OpticksHub::configure
----------------------

Invoked from OpticksHub::init

**/

void OpticksHub::configure()
{
    LOG(LEVEL) << "[" ; 
    m_composition->addConfig(m_umbrella_cfg);  // m_umbrella_cfg collects the BCfg subclass objects such as ViewCfg,CameraCfg etc.. from Composition 

    if(m_ok->has_arg("--dbgcfg")) m_umbrella_cfg->dumpTree(); 

    int argc    = m_ok->getArgc();
    char** argv = m_ok->getArgv();

    LOG(debug) << "argv0 " << argv[0] ; 

    m_umbrella_cfg->commandline(argc, argv);
    m_ok->configure();        // <--- dont like 

    if(m_fcfg->hasError())
    {
        LOG(fatal) << "parse error " << m_fcfg->getErrorMessage() ; 
        m_fcfg->dump("OpticksHub::config m_fcfg");
        m_ok->setExit(true);
        return ; 
    }

#ifdef WITH_M_GLTF
    m_gltf =  m_ok->getGLTF() ;
#endif

    LOG(LEVEL)
          << " argc " << argc 
          << " argv[0] " << ( argv[0] ? argv[0] : "-" )
          << " is_tracer " << m_ok->isTracer() ; 
          ;

    //assert( m_ok->isTracer() ) ; 


    bool compute = m_ok->isCompute();
    bool compute_opt = hasOpt("compute") ;
    if(compute && !compute_opt)
        LOG(error) << "FORCED COMPUTE MODE : as remote session detected " ;  


    if(hasOpt("idpath")) std::cout << m_ok->getIdPath() << std::endl ;
    if(hasOpt("help"))   std::cout << m_umbrella_cfg->getDesc()     << std::endl ;
    if(hasOpt("help|version|idpath"))
    {
        m_ok->setExit(true);
        return ; 
    }


    if(!m_ok->isValid())
    {
        // defer death til after getting help
        LOG(fatal) << "OpticksHub::configure OPTICKS INVALID : missing envvar or geometry path ?" ;
        assert(0);
    }

    LOG(LEVEL) << "]" ; 
}





void OpticksHub::configureServer()
{
#ifdef OPTICKS_NPYSERVER

    m_delegate    = new numpydelegate ; 
    add(new numpydelegateCfg<numpydelegate>("numpydelegate", m_delegate, false));

    if(!hasOpt("nonet"))
    {
      // MAYBE liveConnect should happen in initialization, not here now that event creation happens latter 
        m_delegate->liveConnect(m_umbrella_cfg); // hookup live config via UDP messages

        try { 
            m_server = new numpyserver<numpydelegate>(m_delegate); // connect to external messages 
        } 
        catch( const std::exception& e)
        {
            LOG(fatal) << "OpticksHub::configureServer EXCEPTION " << e.what() ; 
            LOG(fatal) << "OpticksHub::configureServer FAILED to instanciate numpyserver : probably another instance is running : check debugger sessions " ;
        }
    }
#endif
}

void OpticksHub::configureCompositionSize()
{
    assert( m_ok->isConfigured() ); 

    glm::uvec4 size = m_ok->getSize();
    glm::uvec4 position = m_ok->getPosition() ;

    LOG(debug) << "OpticksHub::configureCompositionSize"
              << " size " << gformat(size)
              << " position " << gformat(position)
              ;

    m_composition->setSize( size );
    m_composition->setFramePosition( position );

    unsigned cameratype = m_ok->getCameraType(); 
    m_composition->setCameraType( cameratype ); 

}


/**
OpticksHub::configureState
----------------------------

Invoked from oglrap/OpticksViz.cc

TODO:

Extracate the bookmarks, move to Composition ? 
they dont need to be together with geometry
OpticksHub should only be for geometry needing things.


**/


void OpticksHub::configureState(NConfigurable* scene)
{
    // NState manages the state (in the form of strings) of a collection of NConfigurable objects
    // this needs to happen after configuration and the scene is created

    m_state = m_ok->getState();  
    m_state->setVerbose(false);


    m_state->addConfigurable(scene);
    m_composition->addConstituentConfigurables(m_state); // constituents: trackball, view, camera, clipper

    const char* dir = m_state->getDir();

    LOG(LEVEL)
        << m_state->description()
        << " dir " << dir
        ;

    m_bookmarks   = new Bookmarks(dir) ; 
    m_bookmarks->setState(m_state);
    m_bookmarks->setVerbose();
    m_bookmarks->setInterpolatedViewPeriod(m_fcfg->getInterpolatedViewPeriod());


    m_flightpath = new FlightPath(m_ok->getFlightPathDir()) ; 
    m_flightpath->setCtrl(m_ctrl) ; 


    m_composition->setBookmarks(m_bookmarks);
    m_composition->setFlightPath(m_flightpath); 


    m_composition->setOrbitalViewPeriod(m_fcfg->getOrbitalViewPeriod()); 
    m_composition->setAnimatorPeriod(m_fcfg->getAnimatorPeriod()); 

}

/**
OpticksHub::configureLookupA
-----------------------------

Invoked in init, but only in legacy mode. 

However the lookup remains essenstial for genstep collection
in non-legacy running to translate raw Geant4 material indices 
into GPU texture lines.

This means non-legacy must call OpticksHub::overrideMaterialMapA 
before collecting any gensteps or closing/cross-referencing
the lookup.

Contrast with embedded mode which does not use the hub, see 
G4Opticks::setupMaterialLookup


This was trying and failing to load from 
   /home/blyth/local/opticks/opticksdata/export/OKX4Test/ChromaMaterialMap.json
in direct mode when everything should be from geocache ?


**/

void OpticksHub::configureLookupA()
{
    const char* path = m_ok->getMaterialMap();   // eg "/home/blyth/local/opticks/opticksdata/export/CerenkovMinimal/ChromaMaterialMap.json"
    const char* prefix = m_ok->getMaterialPrefix(); 

    LOG(info)
        << " loading genstep material index map "
        << " path " << path
        << " prefix " << prefix
        ;

    std::map<std::string, unsigned> A ; 
    BMap<std::string, unsigned int>::load(&A, path ); 

    m_lookup->setA(A, prefix, path);
}

/**
OpticksHub::overrideMaterialMapA
---------------------------------

Used from CG4::postinitializeMaterialLookup which gets the
material to int mapping from the Geant4 material table.

**/

void OpticksHub::overrideMaterialMapA(const std::map<std::string, unsigned>& A, const char* msg)
{
    m_lookup->setA( A, "", msg);
}

void OpticksHub::overrideMaterialMapA(const char* jsonA )
{
    m_lookup->setA( jsonA );
}






NCSG* OpticksHub::findEmitter() const  
{
    return m_geotest == NULL ? NULL : m_geotest->findEmitter() ; 
}


GGeoTest* OpticksHub::getGGeoTest()
{
    return m_geotest ; 
}


/**
OpticksHub::getTransform
-------------------------

glm::mat4 OpticksHub::getTransform(int index) const 
{
    return m_ggeo->getTransform(index); 
}

**/


/**
OpticksHub::setupCompositionTargetting
---------------------------------------

Called for example from:

1. oglrap/OpticksViz::uploadGeometry after geometry uploaded
2. okop/OpTracer::render prior to the first trace

**/

void OpticksHub::setupCompositionTargetting()
{
    m_aim->setupCompositionTargetting();
}
void OpticksHub::target()   // point composition at geocenter or the m_evt (last created)
{
    m_aim->target();
}
void OpticksHub::setTarget(unsigned target, bool aim)
{
    m_aim->setTarget(target, aim);
}
unsigned OpticksHub::getTarget()
{
    return m_aim->getTarget();
}
 



void OpticksHub::anaEvent(OpticksEvent* evt)
{
    if(!OpticksEvent::CanAnalyse(evt)) return ; 

    if(m_geotest)
    {
        m_geotest->anaEvent( evt );  
    }
    else
    {
        m_ggeo->anaEvent( evt ); 
    } 
}

void OpticksHub::anaEvent()
{
    LOG(LEVEL) << "[" ;

    OpticksEvent* evt = m_run->getEvent();
    anaEvent(evt); 

    OpticksEvent* g4evt = m_run->getG4Event();
    anaEvent(g4evt); 

    m_run->anaEvent();

    LOG(LEVEL) << "]" ;
}






// from OpticksGen : needed by CGenerator
unsigned        OpticksHub::getSourceCode() const {         return m_gen->getSourceCode() ; }

NPY<float>*     OpticksHub::getInputPhotons() const    {    return m_gen->getInputPhotons() ; }
NPY<float>*     OpticksHub::getInputGensteps() const {      return m_gen->getInputGensteps(); }

TorchStepNPY*   OpticksHub::getTorchstep() const {          return m_gen->getTorchstep() ; }
GenstepNPY*     OpticksHub::getGenstepNPY() const  {        return m_gen->getGenstepNPY() ; }

std::string     OpticksHub::getG4GunConfig() const {        return m_gen->getG4GunConfig(); } 





bool OpticksHub::hasOpt(const char* name)
{
    return m_fcfg->hasOpt(name);
}
bool OpticksHub::isCompute()
{
    return m_ok->isCompute();
}
std::string OpticksHub::getCfgString()
{
    return m_umbrella_cfg->getDescString();
}
OpticksCfg<Opticks>* OpticksHub::getCfg()
{
    return m_fcfg ; 
}





GGeoBase* OpticksHub::getGGeoBaseTest() const 
{
    return m_geotest ? dynamic_cast<GGeoBase*>(m_geotest) : NULL ; 
}



/**
OpticksHub::getGGeoBasePrimary
-------------------------------

Formerly this returned downcast m_gscene (Ana) or m_ggeo (Tri) depending on --gltf option.
Following ana/tri unification within GGeo some years ago, this now always returns downcast  m_ggeo 

**/

GGeoBase* OpticksHub::getGGeoBasePrimary() const 
{
    GGeoBase* ggb = dynamic_cast<GGeoBase*>(m_ggeo) ; 
    return ggb ; 
}
GGeoBase* OpticksHub::getGGeoBase() const   //  2-way : m_geotest/m_ggeo
{
    return m_geotest ? dynamic_cast<GGeoBase*>(m_geotest) : getGGeoBasePrimary() ; 
}

GGeo* OpticksHub::getGGeo() const 
{
    return m_ggeo ; 
}









NState* OpticksHub::getState()
{
    return m_state ; 
}
NLookup* OpticksHub::getLookup()
{
    return m_lookup ; 
}


Opticks* OpticksHub::getOpticks()
{
    return m_ok ; 
}
Composition* OpticksHub::getComposition()
{
    return m_composition ;  
}


Bookmarks* OpticksHub::getBookmarks() const 
{
    return m_bookmarks ; 
}
FlightPath* OpticksHub::getFlightPath() const 
{
    return m_flightpath ; 
}




OpticksGen* OpticksHub::getGen()
{
    return m_gen ; 
}
OpticksRun* OpticksHub::getRun()
{
    return m_run ; 
}



BCfg* OpticksHub::getUmbrellaCfg() const 
{
    return m_umbrella_cfg ; 
}

void OpticksHub::add(BCfg* cfg)
{
    m_umbrella_cfg->add(cfg); 
}



NPY<unsigned char>* OpticksHub::getColorBuffer()
{
    OpticksColors* colors = m_ok->getColors();

    nuvec4 cd = colors->getCompositeDomain() ; 
    glm::uvec4 cd_(cd.x, cd.y, cd.z, cd.w );
  
    m_composition->setColorDomain(cd_); 

    return colors->getCompositeBuffer() ;
}






void OpticksHub::configureEvent(OpticksEvent* evt)
{
    if(!evt) return 

#ifdef OPTICKS_NPYSERVER
    if(m_delegate)
    {
        m_delegate->setEvent(evt); // allows delegate to update evt when NPY messages arrive, hmm locking needed ?
    }
#endif

    m_composition->setEvt(evt);  // look like used only for Composition::setPickPhoton  TODO: reposition this 
    m_composition->setTrackViewPeriod(m_fcfg->getTrackViewPeriod()); 

    NPY<float>* track = evt->loadGenstepDerivativeFromFile("track");
    m_composition->setTrack(track);
}



void OpticksHub::createEvent(unsigned tagoffset)
{
    m_run->createEvent(tagoffset);
}
OpticksEvent* OpticksHub::getG4Event()
{
    return m_run->getG4Event() ; 
}
OpticksEvent* OpticksHub::getEvent()
{
    return m_run->getEvent() ; 
}





OpticksAttrSeq* OpticksHub::getFlagNames()
{
    return m_ok->getFlagNames();
}



void OpticksHub::cleanup()
{
#ifdef OPTICKS_NPYSERVER
    if(m_server) m_server->stop();
#endif

    LOG(LEVEL) << "OpticksHub::cleanup" ; 
    if(m_ok->isGPUMon())
    {
        const char* path = m_ok->getGPUMonPath(); 
        LOG(error) << "GPUMon saving to " << path  ; 
        NGPU* gpu = NGPU::GetInstance() ;
        gpu->saveBuffer(path);
        gpu->dump();
    }  
}


