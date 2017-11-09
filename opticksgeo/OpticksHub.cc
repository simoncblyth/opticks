// sysrap-
#include "SLog.hh"

// brap-
#include "BCfg.hh"
#include "BMap.hh"

#include "NState.hpp"
#include "NLookup.hpp"
#include "NPY.hpp"
#include "NGLM.hpp"
#include "NGLMExt.hpp"

#define GLMVEC4(g) glm::vec4((g).x,(g).y,(g).z,(g).w) 


#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
//#include "NParameters.hpp"

// npy-
#include "Timer.hpp"
#include "TorchStepNPY.hpp"
#include "G4StepNPY.hpp"
#include "Index.hpp"


// numpyserver-
#ifdef WITH_NPYSERVER
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
#include "GGeoTestConfig.hh"
#include "GGeoTest.hh"

//#include "GSurLib.hh"

// okc-
#include "Bookmarks.hh"
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
#include "OpticksGun.hh"
#include "OpticksRun.hh"
#include "OpticksAim.hh"
#include "OpticksGeometry.hh"

#include "PLOG.hh"


//
// **OpticksHub**
//    Non-viz, hostside intersection of config, geometry and event
//    
//    this means is usable from anywhere, so can mop up config
//

OpticksHub::OpticksHub(Opticks* ok) 
   :
   m_log(new SLog("OpticksHub::OpticksHub")),
   m_ok(ok),
   m_gltf(-1),        // m_ok not yet configured, so defer getting the settings
   m_run(m_ok->getRun()),
   m_geometry(NULL),
   m_ggeo(NULL),
   m_gscene(NULL),
   m_composition(new Composition),
#ifdef WITH_NPYSERVER
   m_delegate(NULL),
   m_server(NULL)
#endif
   m_cfg(new BCfg("umbrella", false)),
   m_fcfg(m_ok->getCfg()),
   m_state(NULL),
   m_lookup(new NLookup()),
   m_bookmarks(NULL),
   m_gen(NULL),
   m_gun(NULL),
   m_aim(NULL),
   m_geotest(NULL)
   //m_gsurlib(NULL)

{
   init();
   (*m_log)("DONE");
}


void OpticksHub::init()
{
    add(m_fcfg);

    configure();
    configureServer();
    configureCompositionSize();
    configureLookupA();

    m_aim = new OpticksAim(this) ; 

    loadGeometry() ;    
    configureGeometry() ;    

    m_gen = new OpticksGen(this) ;
    m_gun = new OpticksGun(this) ;

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
       << " m_gun " << m_gun
       ;  

    return ss.str();
}


void OpticksHub::configure()
{
    m_composition->addConfig(m_cfg); 
    //m_cfg->dumpTree();

    int argc    = m_ok->getArgc();
    char** argv = m_ok->getArgv();

    LOG(debug) << "OpticksHub::configure " << argv[0] ; 

    m_cfg->commandline(argc, argv);
    m_ok->configure();      

    if(m_fcfg->hasError())
    {
        LOG(fatal) << "OpticksHub::config parse error " << m_fcfg->getErrorMessage() ; 
        m_fcfg->dump("OpticksHub::config m_fcfg");
        m_ok->setExit(true);
        return ; 
    }

    m_gltf =  m_ok->getGLTF() ;
    LOG(info) << "OpticksHub::configure"
              << " m_gltf " << m_gltf 
              ;

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
}


void OpticksHub::configureServer()
{
#ifdef WITH_NPYSERVER

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
    glm::uvec4 size = m_ok->getSize();
    glm::uvec4 position = m_ok->getPosition() ;

    LOG(debug) << "OpticksHub::configureCompositionSize"
              << " size " << gformat(size)
              << " position " << gformat(position)
              ;

    m_composition->setSize( size );
    m_composition->setFramePosition( position );
}

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

    m_composition->setBookmarks(m_bookmarks);

    m_composition->setOrbitalViewPeriod(m_fcfg->getOrbitalViewPeriod()); 
    m_composition->setAnimatorPeriod(m_fcfg->getAnimatorPeriod()); 
}

void OpticksHub::configureLookupA()
{
    const char* path = m_ok->getMaterialMap(); 
    const char* prefix = m_ok->getMaterialPrefix(); 

    LOG(debug) << "OpticksHub::configureLookupA"
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

    LOG(info) << "OpticksHub::loadGeometry START" ; 

    m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 

    m_geometry->loadGeometry();   


    //   Lookup A and B are now set ...
    //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
    //      B : on GGeo loading in GGeo::setupLookup

    m_ggeo = m_geometry->getGGeo();
    m_gscene = m_ggeo->getScene();

    if(m_ok->isTest())
    {
        LOG(info) << "OpticksHub::loadGeometry --test modifying geometry" ; 

        assert(m_geotest == NULL);

        GGeoBase* basis = getGGeoBasePrimary(); // ana OR tri depending on --gltf

        m_geotest = createTestGeometry(basis);
    }
    else
    {
        LOG(info) << "OpticksHub::loadGeometry NOT modifying geometry" ; 
    }

    registerGeometry();


    m_ggeo->setComposition(m_composition);

    LOG(info) << "OpticksHub::loadGeometry DONE" ; 
}




GGeoTest* OpticksHub::createTestGeometry(GGeoBase* basis)
{
    assert(m_ok->isTest());

    LOG(info) << "OpticksHub::createTestGeometry START" ;

    GGeoTest* testgeo = new GGeoTest(m_ok, basis);

    LOG(info) << "OpticksHub::createTestGeometry DONE" ;

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
        if(transform) vt = glm::make_mat4(transform) ;
    }    
    return vt ;  
}





void OpticksHub::registerGeometry()
{
    LOG(fatal) << "OpticksHub::registerGeometry" ; 
    GMergedMesh* mm0 = getMergedMesh(0);
    assert(mm0);
    m_aim->registerGeometry( mm0 );
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
 






void OpticksHub::configureGeometry()
{
    if(m_ok->isTest())
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
    int restrict_mesh = m_ok->getRestrictMesh() ;  
    int nmm = m_ggeo->getNumMergedMesh();

    LOG(info) << "OpticksHub::configureGeometryTri" 
              << " restrict_mesh " << restrict_mesh
              << " nmm " << nmm
              ;

    for(int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i);
        if(!mm) continue ; 
        if(restrict_mesh > -1 && i != restrict_mesh ) mm->setGeoCode(OpticksConst::GEOCODE_SKIP);      
    }
}

void OpticksHub::configureGeometryTriAna()
{
    int restrict_mesh = m_ok->getRestrictMesh() ;  

    LOG(info) << "OpticksHub::configureGeometryTriAna" 
              << " restrict_mesh " << restrict_mesh
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

        if(restrict_mesh > -1 && i != restrict_mesh ) 
        {
            mm_a->setGeoCode(OpticksConst::GEOCODE_SKIP);      
            mm_t->setGeoCode(OpticksConst::GEOCODE_SKIP);      
        }
    }
}

void OpticksHub::configureGeometryTest()
{
    int restrict_mesh = m_ok->getRestrictMesh() ;  
    GGeoBase* ggb = getGGeoBase();   // either ana or tri depending on gltf
    GGeoLib*  lib = ggb->getGeoLib() ; 
    int nmm = lib->getNumMergedMesh();

    for(int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = lib->getMergedMesh(i);
        assert( mm );  

        if(restrict_mesh > -1 && i != restrict_mesh ) 
        {
            mm->setGeoCode(OpticksConst::GEOCODE_SKIP);      
        }
    }
    // actually unlikely to need restrictmesh with --test 

}



void OpticksHub::anaEvent()
{
    LOG(info) << "OpticksHub::anaEvent" ;

    OpticksEvent* evt = m_run->getEvent();

    // hmm add anaEvent to GGeoBase ?

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


    m_run->anaEvent();
}





std::string OpticksHub::getG4GunConfig()
{
    return m_gun->getConfig();
}

TorchStepNPY* OpticksHub::getTorchstep()   // needed by CGenerator
{
    return m_gen->getTorchstep() ; 
}

GenstepNPY* OpticksHub::getGenstepNPY()   // needed by CGenerator
{
    return m_gen->getGenstepNPY() ; 
}


NPY<float>* OpticksHub::getInputPhotons()   // needed by CGenerator
{
    return m_gen->getInputPhotons() ; 
}

unsigned OpticksHub::getSourceCode() const 
{
    return m_gen->getSourceCode() ; 
}

NPY<float>* OpticksHub::getInputGensteps() const 
{
    return m_gen->getInputGensteps();
}

/*
NPY<float>* OpticksHub::getEmitterGensteps() const 
{
    return m_gen->getEmitterGensteps();
}
*/


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




GGeo* OpticksHub::getGGeo()
{
    return m_ggeo ; 
}
GGeoLib* OpticksHub::getGeoLib()
{
    return m_ggeo->getGeoLib() ; 
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
    LOG(info) << "OpticksHub::getGGeoBasePrimary"
              << " analytic switch  "
              << " m_gltf " << m_gltf
              << " ggb " << ( ggb ? ggb->getIdentifier() : "NULL" )
               ;

    return ggb ; 
}
GGeoBase* OpticksHub::getGGeoBase() const 
{
    return m_geotest ? dynamic_cast<GGeoBase*>(m_geotest) : getGGeoBasePrimary() ; 
}




GMergedMesh* OpticksHub::getMergedMesh( unsigned index )
{
    GGeoBase* ggb = getGGeoBase();  // 3-way
    return ggb->getMergedMesh(index);
}
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



/*
GSurLib* OpticksHub::createSurLib(GGeoBase* ggb)  // KILL
{
    GSurLib* gsl = new GSurLib(m_ok, ggb ); 
    return gsl ; 
}

GSurLib* OpticksHub::getSurLib() // KILL
{ 
    if( m_gsurlib == NULL )
    {
        // this method motivating making GGeoTest into a GGeoBase : ie standard geo provider
        GGeoBase* ggb = getGGeoBase();    // three-way choice 
        m_gsurlib = createSurLib(ggb) ;
    }
    return m_gsurlib ; 
}
*/



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
Bookmarks* OpticksHub::getBookmarks()
{
    return m_bookmarks ; 
}

Timer* OpticksHub::getTimer()
{
    OpticksEvent* evt = m_run->getEvent();
    return evt ? evt->getTimer() : m_ok->getTimer() ; 
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

#ifdef WITH_NPYSERVER
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
OpticksAttrSeq* OpticksHub::getMaterialNames()
{
    return m_geometry->getMaterialNames();
}
OpticksAttrSeq* OpticksHub::getBoundaryNames()
{
    return m_geometry->getBoundaryNames();
}
std::map<unsigned int, std::string> OpticksHub::getBoundaryNamesMap()
{
    return m_geometry->getBoundaryNamesMap();
}

void OpticksHub::cleanup()
{
#ifdef WITH_NPYSERVER
    if(m_server) m_server->stop();
#endif
}




void OpticksHub::dumpSolids(unsigned cursor, GMergedMesh* mm, const char* msg )  
{
    assert( mm );
    unsigned num_solids = mm->getNumSolids();

    LOG(info) << "OpticksHub::dumpSolids "
              << msg 
              << " num_solids " << num_solids 
              ;

    bool test = m_ok->isTest() ; 

    GNodeLib* nodelib = getNodeLib();
    for(unsigned i=0 ; i < std::min(num_solids, 20u) ; i++)
    {
         glm::vec4 ce_ = mm->getCE(i);
         std::cout << " " << std::setw(3) << i 
                   << " " << ( i == cursor ? "**" : "  " ) 
                   << std::setw(50) << ( test ? "test" : nodelib->getLVName(i) )
                   << " " 
                   << gpresent( "ce", ce_ )
                   ;
    }
}




