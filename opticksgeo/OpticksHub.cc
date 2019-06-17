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
#include "GScene.hh"
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
#include "OpticksGeometry.hh"

#include "PLOG.hh"

const plog::Severity OpticksHub::LEVEL = debug ; 
//const plog::Severity OpticksHub::LEVEL = error ; 


//  hmm : the hub could be a GGeoBase ?

const char* OpticksHub::getIdentifier()
{
    GGeoBase* ggb = getGGeoBase();  // 3-way
    return ggb->getIdentifier();
}
GMergedMesh* OpticksHub::getMergedMesh( unsigned index )
{
    GGeoBase* ggb = getGGeoBase();  // 3-way   m_geotest/m_ggeo/m_gscene
    return ggb->getMergedMesh(index);
}

/*
GPmtLib* OpticksHub::getPmtLib()
{
    GGeoBase* ggb = getGGeoBase();  // 3-way
    return ggb->getPmtLib();
}
*/


GNodeLib* OpticksHub::getNodeLib()
{
    GGeoBase* ggb = getGGeoBase();  // 3-way
    return ggb->getNodeLib();
}
GMaterialLib* OpticksHub::getMaterialLib()
{  
    GGeoBase* ggb = getGGeoBase();  // 3-way
    return ggb->getMaterialLib() ; 
}
GSurfaceLib* OpticksHub::getSurfaceLib() 
{   
    GGeoBase* ggb = getGGeoBase();  // 3-way
    return ggb->getSurfaceLib() ; 
}

GBndLib* OpticksHub::getBndLib() 
{   
    GGeoBase* ggb = getGGeoBase();  // 3-way
    return ggb->getBndLib() ; 
}
GScintillatorLib* OpticksHub::getScintillatorLib() 
{ 
    GGeoBase* ggb = getGGeoBase();  // 3-way
    return ggb->getScintillatorLib() ;
}
GSourceLib* OpticksHub::getSourceLib() 
{ 
    GGeoBase* ggb = getGGeoBase();  // 3-way
    return ggb->getSourceLib() ;
}
GGeoLib* OpticksHub::getGeoLib()
{
    GGeoBase* ggb = getGGeoBase();  // 3-way
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
   m_gltf(-1),        // m_ok not yet configured, so defer getting the settings
   m_run(m_ok->getRun()),
   m_geometry(NULL),
   m_ggeo(GGeo::GetInstance()),   // if there is a GGeo instance already extant adopt it, otherwise load one  
   m_gscene(NULL),
   m_composition(new Composition),
#ifdef OPTICKS_NPYSERVER
   m_delegate(NULL),
   m_server(NULL)
#endif
   m_cfg(new BCfg("umbrella", false)),
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
    pLOG(LEVEL,0) << "[" ;   // -1 : one notch more easily seen than LEVEL

    //m_composition->setCtrl(this); 

    add(m_fcfg);

    configure();
    // configureGeometryPrep();
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
    if(m_err) return ; 



    configureGeometry() ;    

    deferredGeometryPrep(); 



    m_gen = new OpticksGen(this) ;

    pLOG(LEVEL,0) << "]" ; 
}



std::string OpticksHub::desc() const 
{
    std::stringstream ss ; 

    GGeoBase* ggb = getGGeoBase(); 

    ss << "OpticksHub"
       << " encumbent " << ( ggb ? ggb->getIdentifier() : "-" ) 
       << " m_ggeo " << m_ggeo
       << " m_gscene " << m_gscene
       << " m_geometry " << m_geometry
       << " m_gen " << m_gen
       ;  

    return ss.str();
}


void OpticksHub::configure()
{
    LOG(LEVEL) << "[" ; 
    m_composition->addConfig(m_cfg); 
    //m_cfg->dumpTree();

    int argc    = m_ok->getArgc();
    char** argv = m_ok->getArgv();

    LOG(debug) << "argv0 " << argv[0] ; 

    m_cfg->commandline(argc, argv);
    m_ok->configure();        // <--- dont like 

    if(m_fcfg->hasError())
    {
        LOG(fatal) << "parse error " << m_fcfg->getErrorMessage() ; 
        m_fcfg->dump("OpticksHub::config m_fcfg");
        m_ok->setExit(true);
        return ; 
    }

    m_gltf =  m_ok->getGLTF() ;

    LOG(LEVEL)
          << " argc " << argc 
          << " argv[0] " << ( argv[0] ? argv[0] : "-" )
          << " m_gltf " << m_gltf 
          << " is_tracer " << m_ok->isTracer() ; 
          ;

    //assert( m_ok->isTracer() ) ; 


    bool compute = m_ok->isCompute();
    bool compute_opt = hasOpt("compute") ;
    if(compute && !compute_opt)
        LOG(warning) << "OpticksHub::configure FORCED COMPUTE MODE : as remote session detected " ;  


    if(hasOpt("idpath")) std::cout << m_ok->getIdPath() << std::endl ;
    if(hasOpt("help"))   std::cout << m_cfg->getDesc()     << std::endl ;
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




/*
void OpticksHub::configureGeometryPrep()
{
    bool geocache = !m_fcfg->hasOpt("nogeocache") ;
    bool instanced = !m_fcfg->hasOpt("noinstanced") ; // find repeated geometry 

    LOG(debug) << "OpticksGeometry::init"
              << " geocache " << geocache 
              << " instanced " << instanced
              ;

    m_ok->setGeocache(geocache);
    m_ok->setInstanced(instanced); // find repeated geometry 
}

*/



void OpticksHub::configureServer()
{
#ifdef OPTICKS_NPYSERVER

    m_delegate    = new numpydelegate ; 
    add(new numpydelegateCfg<numpydelegate>("numpydelegate", m_delegate, false));

    if(!hasOpt("nonet"))
    {
      // MAYBE liveConnect should happen in initialization, not here now that event creation happens latter 
        m_delegate->liveConnect(m_cfg); // hookup live config via UDP messages

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

    LOG(fatal) << "OpticksHub::configureState " 
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

Invoked in init 

This is trying and failing to load from 
   /home/blyth/local/opticks/opticksdata/export/OKX4Test/ChromaMaterialMap.json
in direct mode when everything should be from geocache ?

**/

void OpticksHub::configureLookupA()
{
    const char* path = m_ok->getMaterialMap();   // eg "/home/blyth/local/opticks/opticksdata/export/CerenkovMinimal/ChromaMaterialMap.json"
    const char* prefix = m_ok->getMaterialPrefix(); 

    LOG(debug)
        << " loading genstep material index map "
        << " path " << path
        << " prefix " << prefix
        ;

    std::map<std::string, unsigned> A ; 
    BMap<std::string, unsigned int>::load(&A, path ); 

    m_lookup->setA(A, prefix, path);
}

void OpticksHub::overrideMaterialMapA(const std::map<std::string, unsigned>& A, const char* msg)
{
   // Used from OKG4Mgr to override the default mapping when using G4 steps directly 
    m_lookup->setA( A, "", msg);
}

void OpticksHub::overrideMaterialMapA(const char* jsonA )
{
    m_lookup->setA( jsonA );
}



void OpticksHub::loadGeometry()
{
    assert(m_geometry == NULL && "OpticksHub::loadGeometry should only be called once");

    LOG(info) << "[ " << m_ok->getIdPath()  ; 

    m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 

    m_geometry->loadGeometry();   

    m_ggeo = m_geometry->getGGeo();

    m_gscene = m_ggeo->getScene();


    //   Lookup A and B are now set ...
    //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
    //      B : on GGeo loading in GGeo::setupLookup


    if(m_ok->isTest())  // --test : instanciate GGeoTest 
    {
        LOG(info) << "--test modifying geometry" ; 

        assert(m_geotest == NULL);

        GGeoBase* basis = getGGeoBasePrimary(); // ana OR tri depending on --gltf

        m_geotest = createTestGeometry(basis);

        int err = m_geotest->getErr() ;
        if(err) 
        {
            setErr(err);
            return ; 
        }
    }
    else
    {
        LOG(LEVEL) << "NOT modifying geometry" ; 
    }

    registerGeometry();

    m_ggeo->setComposition(m_composition);

    m_ggeo->close();  // mlib and slib  (June 2018, following remove the auto-trigger-close on getIndex in the proplib )

    LOG(info) << "]" ; 
}



void OpticksHub::adoptGeometry()
{
    LOG(LEVEL) << "[" ; 
    assert( m_ggeo ); 
    assert( m_ggeo->isPrepared() && "MUST GGeo::prepare() before geometry can be adopted, and uploaded to GPU " ) ;

    m_gscene = m_ggeo->getScene();  // DONT LIKE SEPARATE GScene 

    registerGeometry();

    m_ggeo->setComposition(m_composition);

    LOG(LEVEL) << "]" ; 
}



GGeoTest* OpticksHub::createTestGeometry(GGeoBase* basis)
{
    assert(m_ok->isTest());  // --test  : instanciate GGeoTest using the basis

    LOG(info) << "[" ;

    GGeoTest* testgeo = new GGeoTest(m_ok, basis);

    LOG(info) << "]" ;

    return testgeo ; 
}


NCSG* OpticksHub::findEmitter() const  
{
    return m_geotest == NULL ? NULL : m_geotest->findEmitter() ; 
}


GGeoTest* OpticksHub::getGGeoTest()
{
    return m_geotest ; 
}


glm::mat4 OpticksHub::getTransform(int index)
{
    glm::mat4 vt ; 
    if(index > -1)
    {    
        GMergedMesh* mesh0 = getMergedMesh(0);
        float* transform = mesh0 ? mesh0->getTransform(index) : NULL ;
        assert( transform ) ; 

        if(transform) vt = glm::make_mat4(transform) ;
    }    
    return vt ;  
}





void OpticksHub::registerGeometry()
{
    LOG(fatal) << "[" ; 

    const char* ggb = getIdentifier(); 
    LOG(fatal) << " ggb " << ggb ;  
    GMergedMesh* mm0 = getMergedMesh(0);
    assert(mm0);
    m_aim->registerGeometry( mm0 );

    LOG(fatal) << "]" ; 
}

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
 




/**
OpticksHub::configureGeometry
------------------------------

TODO: 
   see if can eliminate the tri/ana mess now, 
   following adoption of unified tri+ana approach 
   ... where the assumption is to always have both 
   ... and then switch as picked by options 
       at late stage (in OGeo) 

   configureGeometryTri
   configureGeometryTriAna
        just setting geocode
        BUT --xanalytic  isXAnalytic may trump this 
      
        opticks-if xanalytic
              
   Better to set the geocode in one place only... 
   close to where they are used in OGeo ? 

**/

void OpticksHub::configureGeometry()
{
    if(m_ok->isTest())  // --test : configure mesh skips  
    {
        configureGeometryTest();
    }
    else if(m_gltf==0) 
    { 
        configureGeometryTri();
    }
    else
    {
        configureGeometryTriAna();
    }
}


void OpticksHub::configureGeometryTri()
{
    int nmm = m_ggeo->getNumMergedMesh();

    LOG(LEVEL) 
              << "setting geocode" 
              << " nmm " << nmm
              ;

    for(int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i);
        if(!mm) continue ; 

        if(!m_ok->isEnabledMergedMesh(i)) 
        {
            LOG(info) << "setting  OpticksConst::GEOCODE_SKIP for mm " << i ; 
            mm->setGeoCode(OpticksConst::GEOCODE_SKIP);      
        }
    }
}


void OpticksHub::configureGeometryTriAna()
{

    LOG(info) << "OpticksHub::configureGeometryTriAna" 
              << " desc " << desc() 
              ;

    GGeoBase* ana_g = getGGeoBaseAna();   // GScene downcast
    GGeoBase* tri_g = getGGeoBaseTri();   // GGeo downcast

    GGeoLib* ana = ana_g ? ana_g->getGeoLib() : NULL ; 
    GGeoLib* tri = tri_g ? tri_g->getGeoLib() : NULL ; 

    int nmm_a = ana->getNumMergedMesh();
    int nmm_t = tri->getNumMergedMesh();

    bool match = nmm_a == nmm_t ; 
    if(!match)
    {
        LOG(fatal) << "OpticksHub::configureGeometryTriAna"
                   << " MISMATCH "
                   << " nmm_a " << nmm_a 
                   << " nmm_t " << nmm_t
                   ; 
    }

    assert( match );

    for(int i=0 ; i < nmm_a ; i++)
    {
        GMergedMesh* mm_a = ana->getMergedMesh(i);
        GMergedMesh* mm_t = tri->getMergedMesh(i);
        assert( mm_a && mm_t );  

        if(!m_ok->isEnabledMergedMesh(i)) 
        {
            LOG(info) << "setting  OpticksConst::GEOCODE_SKIP for mm " << i ; 
            mm_a->setGeoCode(OpticksConst::GEOCODE_SKIP);      
            mm_t->setGeoCode(OpticksConst::GEOCODE_SKIP);      
        }
    }
}

void OpticksHub::configureGeometryTest()
{
    GGeoBase* ggb = getGGeoBase();   // either ana or tri depending on gltf
    GGeoLib*  lib = ggb->getGeoLib() ; 
    int nmm = lib->getNumMergedMesh();

    for(int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = lib->getMergedMesh(i);
        assert( mm );  
        if(!m_ok->isEnabledMergedMesh(i)) 
        {
            LOG(info) << "setting  OpticksConst::GEOCODE_SKIP for mm " << i ; 
            mm->setGeoCode(OpticksConst::GEOCODE_SKIP);      
        }
    }
    // actually unlikely to need restrictmesh with --test 

}




/**
OpticksHub::deferredGeometryPrep
---------------------------------

Invoked from OpticksHub::init after loading or adopting geometry.


**/

void OpticksHub::deferredGeometryPrep()
{
    m_ggeo->deferredCreateGParts() ;    
}







void OpticksHub::anaEvent(OpticksEvent* evt)
{
    if(!OpticksEvent::CanAnalyse(evt)) return ; 

    if(m_geotest)
    {
        m_geotest->anaEvent( evt );  
    }
    else if(m_gscene)
    { 
        m_gscene->anaEvent( evt ); 
    }
    else
    {
        m_ggeo->anaEvent( evt ); 
    } 
}

void OpticksHub::anaEvent()
{
    LOG(info) << "OpticksHub::anaEvent" ;

    OpticksEvent* evt = m_run->getEvent();
    anaEvent(evt); 

    OpticksEvent* g4evt = m_run->getG4Event();
    anaEvent(g4evt); 

    m_run->anaEvent();
}






// from OpticksGen : needed by CGenerator
unsigned        OpticksHub::getSourceCode() const {         return m_gen->getSourceCode() ; }

NPY<float>*     OpticksHub::getInputPhotons() const    {    return m_gen->getInputPhotons() ; }
NPY<float>*     OpticksHub::getInputGensteps() const {      return m_gen->getInputGensteps(); }
//NPY<float>*     OpticksHub::getInputPrimaries() const  {    return m_gen->getInputPrimaries() ; }

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
    return m_cfg->getDescString();
}
OpticksCfg<Opticks>* OpticksHub::getCfg()
{
    return m_fcfg ; 
}






GGeoBase* OpticksHub::getGGeoBaseAna() const 
{
    return m_gscene ? dynamic_cast<GGeoBase*>(m_gscene) : NULL ; 
}

GGeoBase* OpticksHub::getGGeoBaseTri() const 
{
    return m_ggeo ? dynamic_cast<GGeoBase*>(m_ggeo) : NULL ; 
}

GGeoBase* OpticksHub::getGGeoBaseTest() const 
{
    return m_geotest ? dynamic_cast<GGeoBase*>(m_geotest) : NULL ; 
}

GGeoBase* OpticksHub::getGGeoBasePrimary() const 
{
    GGeoBase* ggb = m_gltf ? dynamic_cast<GGeoBase*>(m_gscene) : dynamic_cast<GGeoBase*>(m_ggeo) ; 

    /*
    LOG(info) << "OpticksHub::getGGeoBasePrimary"
              << " analytic switch  "
              << " m_gltf " << m_gltf
              << " ggb " << ( ggb ? ggb->getIdentifier() : "NULL" )
               ;
    */ 

    return ggb ; 
}
GGeoBase* OpticksHub::getGGeoBase() const   //  3-way : m_geotest/m_gscene/m_ggeo
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
OpticksGeometry* OpticksHub::getGeometry()
{
    return m_geometry ;  
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



void OpticksHub::add(BCfg* cfg)
{
    m_cfg->add(cfg); 
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


void OpticksHub::dumpVolumes(unsigned cursor, GMergedMesh* mm, const char* msg )  
{
    //assert(0); 
    assert( mm );
    unsigned num_volumes = mm->getNumVolumes();

    LOG(info) << "OpticksHub::dumpVolumes "
              << msg 
              << " num_volumes " << num_volumes 
              ;

    bool test = m_ok->isTest() ;    // --test : dumping volumes

    GNodeLib* nodelib = getNodeLib();
    for(unsigned i=0 ; i < std::min(num_volumes, 20u) ; i++)
    {
         glm::vec4 ce_ = mm->getCE(i);
         std::cout << " " << std::setw(7) << i 
                   << " " << ( i == cursor ? "**" : "  " ) 
                   << std::setw(70) << ( test ? "test" : nodelib->getLVName(i) )
                   << " " 
                   << gpresent( "ce", ce_ )
                   ;
    }


    float extent_cut = 5000.f ;
    LOG(info) << " volumes with extent greater than " << extent_cut << " mm " ; 
    for(unsigned i=0 ; i < num_volumes ; i++)
    {
         glm::vec4 ce_ = mm->getCE(i);

         if(ce_.w > extent_cut )
         std::cout << " " << std::setw(7) << i 
                   << " " << ( i == cursor ? "**" : "  " ) 
                   << std::setw(70) << ( test ? "test" : nodelib->getLVName(i) )
                   << " " 
                   << gpresent( "ce", ce_ )
                   ;
    }





}




