
// npy-
#include "Timer.hpp"
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
#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GPmt.hh"
#include "GParts.hh"
#include "GMergedMesh.hh"
#include "GNodeLib.hh"
#include "GGeo.hh"

// assimpwrap
#include "AssimpGGeo.hh"

// openmeshrap-
#include "MFixer.hh"
#include "MTool.hh"


// opticksgeo-
#include "OpticksGeometry.hh"


#include "PLOG.hh"

#define GLMVEC4(g) glm::vec4((g).x,(g).y,(g).z,(g).w) 


// TODO: move to OK_PROFILE 
#define TIMER(s) \
    { \
       if(m_ok)\
       {\
          Timer& t = *(m_ok->getTimer()) ;\
          t((s)) ;\
       }\
    }


OpticksGeometry::OpticksGeometry(OpticksHub* hub)
   :
   m_hub(hub),
   m_ok(m_hub->getOpticks()),
   m_composition(m_hub->getComposition()),
   m_fcfg(m_ok->getCfg()),
   m_ggeo(NULL),
   m_mesh0(NULL),
   m_target(0),
   m_target_deferred(0),
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
    bool geocache = !m_fcfg->hasOpt("nogeocache") ;
    bool instanced = !m_fcfg->hasOpt("noinstanced") ; // find repeated geometry 

    LOG(debug) << "OpticksGeometry::init"
              << " geocache " << geocache 
              << " instanced " << instanced
              ;

    m_ok->setGeocache(geocache);
    m_ok->setInstanced(instanced); // find repeated geometry 

    m_ggeo = new GGeo(m_ok);
    m_ggeo->setLookup(m_hub->getLookup());
}

glm::vec4 OpticksGeometry::getCenterExtent() // TODO: not-tri specific -> move 
{

    if(!m_mesh0)
    {
        LOG(fatal) << "OpticksGeometry::getCenterExtent" 
                   << " mesh0 NULL "
                   ;
        
        return glm::vec4(0.f,0.f,0.f,1.f) ;
    } 


    glm::vec4 mmce = GLMVEC4(m_mesh0->getCenterExtent(0)) ;
    return mmce ; 
}



void  OpticksGeometry::setTarget(unsigned target, bool aim)  
{
    // formerly of oglrap-/Scene
    // invoked by OpticksViz::uploadGeometry OpticksViz::init

   if(m_mesh0 == NULL)
    {    
        LOG(info) << "OpticksGeometry::setTarget " << target << " deferring as geometry not loaded " ; 
        m_target_deferred = target ; 
        return ; 
    }    
    m_target = target ; 


    GNodeLib* nodelib = m_ggeo->getNodeLib();

    unsigned num_solids = m_mesh0->getNumSolids();

    LOG(info) << "OpticksGeometry::setTarget"
              << " num_solids " << num_solids 
              ;
    for(unsigned i=0 ; i < std::min(num_solids, 20u) ; i++)
    {
         glm::vec4 ce_ = m_mesh0->getCE(i);
         std::cout << " " << std::setw(3) << i 
                   << " " << ( i == target ? "**" : "  " ) 
                   << std::setw(50) << nodelib->getLVName(i)
                   << " " 
                   << gpresent( "ce", ce_ )
                   
                   ;
    }

    glm::vec4 ce = m_mesh0->getCE(target);


    LOG(fatal)<<"OpticksGeometry::setTarget " 
             << " based on CenterExtent from m_mesh0 "
             << " target " << target 
             << " aim " << aim
             << " ce " 
             << " " << ce.x 
             << " " << ce.y 
             << " " << ce.z 
             << " " << ce.w 
             ;    

    m_composition->setCenterExtent(ce, aim); 
}

unsigned OpticksGeometry::getTargetDeferred()
{
    return m_target_deferred ;
}
unsigned OpticksGeometry::getTarget()
{
    return m_target ;
}






OpticksAttrSeq* OpticksGeometry::getMaterialNames()
{
     OpticksAttrSeq* qmat = m_ggeo->getMaterialLib()->getAttrNames();
     qmat->setCtrl(OpticksAttrSeq::SEQUENCE_DEFAULTS);
     return qmat ; 
}

OpticksAttrSeq* OpticksGeometry::getBoundaryNames()
{
     GBndLib* blib = m_ggeo->getBndLib();
     OpticksAttrSeq* qbnd = blib->getAttrNames();
     if(!qbnd->hasSequence())
     {    
         blib->close();
         assert(qbnd->hasSequence());
     }    
     qbnd->setCtrl(OpticksAttrSeq::VALUE_DEFAULTS);
     return qbnd ;
}

std::map<unsigned int, std::string> OpticksGeometry::getBoundaryNamesMap()
{
    OpticksAttrSeq* qbnd = getBoundaryNames() ;
    return qbnd->getNamesMap(OpticksAttrSeq::ONEBASED) ;
}



void OpticksGeometry::loadGeometry()
{
    bool modify = m_ok->hasOpt("test") ;

    LOG(info) << "OpticksGeometry::loadGeometry START, modifyGeometry? " << modify  ; 

    loadGeometryBase(); //  usually from cache

    if(!m_ggeo->isValid())
    {
        LOG(warning) << "OpticksGeometry::loadGeometry finds invalid geometry, try creating geocache with --nogeocache/-G option " ; 
        m_ok->setExit(true); 
        return ; 
    }

    if(modify) modifyGeometry() ;

    // hmm is this modify approach still needed ? perhaps just loadTestGeometry ?
    // probably the issue is GGeo does too much ...


    fixGeometry();

    registerGeometry();

    if(!m_ok->isGeocache())
    {
        LOG(info) << "OpticksGeometry::loadGeometry early exit due to --nogeocache/-G option " ; 
        m_ok->setExit(true); 
    }


    LOG(info) << "OpticksGeometry::loadGeometry DONE " ; 
    TIMER("loadGeometry");
}


void OpticksGeometry::loadGeometryBase()
{
    LOG(info) << "OpticksGeometry::loadGeometryBase START " ; 
    OpticksResource* resource = m_ok->getResource();

    if(m_ok->hasOpt("qe1"))
        m_ggeo->getSurfaceLib()->setFakeEfficiency(1.0);


    m_ggeo->setLoaderImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr

    m_ggeo->setLoaderVerbosity(m_fcfg->getLoaderVerbosity());    

    m_ggeo->setMeshJoinImp(&MTool::joinSplitUnion);
    m_ggeo->setMeshVerbosity(m_fcfg->getMeshVerbosity());    
    m_ggeo->setMeshJoinCfg( resource->getMeshfix() );

    std::string meshversion = m_fcfg->getMeshVersion() ;;
    if(!meshversion.empty())
    {
        LOG(warning) << "OpticksGeometry::loadGeometry using debug meshversion " << meshversion ;  
        m_ggeo->getGeoLib()->setMeshVersion(meshversion.c_str());
    }

    m_ggeo->loadGeometry();   // potentially from cache : for gltf > 0 loads both tri and ana geometry 
        
    if(m_ggeo->getMeshVerbosity() > 2)
    {
        GMergedMesh* mesh1 = m_ggeo->getMergedMesh(1);
        if(mesh1)
        {
            mesh1->dumpSolids("OpticksGeometry::loadGeometryBase mesh1");
            mesh1->save("$TMP", "GMergedMesh", "baseGeometry") ;
        }
    }

    LOG(info) << "OpticksGeometry::loadGeometryBase DONE " ; 
    TIMER("loadGeometryBase");
}

void OpticksGeometry::modifyGeometry()
{
    //assert(m_ok->hasOpt("test"));
    assert( m_ok->isTest() );

    LOG(debug) << "OpticksGeometry::modifyGeometry" ;

    //std::string testconf = m_fcfg->getTestConfig();
    //m_ggeo->modifyGeometry( testconf.empty() ? NULL : testconf.c_str() );

    const char* testconf = m_ok->getTestConfig() ;
    m_ggeo->modifyGeometry( testconf );


    if(m_ggeo->getMeshVerbosity() > 2)
    {
        GMergedMesh* mesh0 = m_ggeo->getMergedMesh(0);
        if(mesh0)
        { 
            mesh0->dumpSolids("OpticksGeometry::modifyGeometry mesh0");
            mesh0->save("$TMP", "GMergedMesh", "modifyGeometry") ;
        }
    }


    TIMER("modifyGeometry"); 
}


void OpticksGeometry::fixGeometry()
{
    if(m_ggeo->isLoaded())
    {
        LOG(debug) << "OpticksGeometry::fixGeometry needs to be done precache " ;
        return ; 
    }
    LOG(info) << "OpticksGeometry::fixGeometry" ; 

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
    TIMER("fixGeometry"); 
}






void OpticksGeometry::registerGeometry()
{
    LOG(debug) << "OpticksGeometry::registerGeometry" ; 

    //for(unsigned int i=1 ; i < m_ggeo->getNumMergedMesh() ; i++) m_ggeo->dumpNodeInfo(i);

    m_mesh0 = m_ggeo->getMergedMesh(0); 

    if(!m_mesh0)
    {
        LOG(error) << "OpticksGeometry::registerGeometry"
                   <<  " NULL mesh0 "
                   ;
        return ;   
    }   


    gfloat4 ce0 = m_mesh0->getCenterExtent(0);  // 0 : all geometry of the mesh, >0 : specific volumes

    m_ok->setSpaceDomain( glm::vec4(ce0.x,ce0.y,ce0.z,ce0.w) );


    LOG(debug) << "OpticksGeometry::registerGeometry setting opticks SpaceDomain : " 
                      << " x " << ce0.x
                      << " y " << ce0.y
                      << " z " << ce0.z
                      << " w " << ce0.w
                      ;
 
    TIMER("registerGeometry"); 
}




