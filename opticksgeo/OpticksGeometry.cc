
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


#define TIMER(s) \
    { \
       if(m_opticks)\
       {\
          Timer& t = *(m_opticks->getTimer()) ;\
          t((s)) ;\
       }\
    }


OpticksGeometry::OpticksGeometry(OpticksHub* hub)
   :
   m_hub(hub),
   m_opticks(m_hub->getOpticks()),
   m_composition(m_hub->getComposition()),
   m_fcfg(m_opticks->getCfg()),
   m_ggeo(NULL),
   m_mesh0(NULL),
   m_target(0),
   m_target_deferred(0)
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

    m_opticks->setGeocache(geocache);
    m_opticks->setInstanced(instanced); // find repeated geometry 

    m_ggeo = new GGeo(m_opticks);
    m_ggeo->setLookup(m_hub->getLookup());
}

glm::vec4 OpticksGeometry::getCenterExtent()
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

   if(m_mesh0 == NULL)
    {    
        LOG(info) << "OpticksGeometry::setTarget " << target << " deferring as geometry not loaded " ; 
        m_target_deferred = target ; 
        return ; 
    }    
    m_target = target ; 

    gfloat4 ce_ = m_mesh0->getCenterExtent(target);

    glm::vec4 ce(ce_.x, ce_.y, ce_.z, ce_.w ); 

    LOG(info)<<"OpticksGeometry::setTarget " 
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
    bool modify = m_opticks->hasOpt("test") ;

    LOG(debug) << "OpticksGeometry::loadGeometry START, modifyGeometry? " << modify  ; 

    loadGeometryBase(); //  usually from cache

    if(!m_ggeo->isValid())
    {
        LOG(warning) << "OpticksGeometry::loadGeometry finds invalid geometry, try creating geocache with --nogeocache/-G option " ; 
        m_opticks->setExit(true); 
        return ; 
    }

    if(modify) modifyGeometry() ;


    fixGeometry();

    registerGeometry();

    if(!m_opticks->isGeocache())
    {
        LOG(info) << "OpticksGeometry::loadGeometry early exit due to --nogeocache/-G option " ; 
        m_opticks->setExit(true); 
    }

    configureGeometry();

    TIMER("loadGeometry");
}


void OpticksGeometry::loadGeometryBase()
{
    OpticksResource* resource = m_opticks->getResource();

    if(m_opticks->hasOpt("qe1"))
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

    m_ggeo->loadGeometry();   // potentially from cache 
        
    if(m_ggeo->getMeshVerbosity() > 2)
    {
        GMergedMesh* mesh1 = m_ggeo->getMergedMesh(1);
        if(mesh1)
        {
            mesh1->dumpSolids("OpticksGeometry::loadGeometryBase mesh1");
            mesh1->save("$TMP", "GMergedMesh", "baseGeometry") ;
        }
    }

    TIMER("loadGeometryBase");
}

void OpticksGeometry::modifyGeometry()
{
    assert(m_opticks->hasOpt("test"));
    LOG(debug) << "OpticksGeometry::modifyGeometry" ;

    std::string testconf = m_fcfg->getTestConfig();
    m_ggeo->modifyGeometry( testconf.empty() ? NULL : testconf.c_str() );


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
    fixer->setVerbose(m_opticks->hasOpt("meshfixdbg"));
    fixer->fixMesh();
 
    bool zexplode = m_opticks->hasOpt("zexplode");
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



void OpticksGeometry::configureGeometry()
{
    int restrict_mesh = m_fcfg->getRestrictMesh() ;  
    int analytic_mesh = m_fcfg->getAnalyticMesh() ; 

    int nmm = m_ggeo->getNumMergedMesh();

    LOG(debug) << "OpticksGeometry::configureGeometry" 
              << " restrict_mesh " << restrict_mesh
              << " analytic_mesh " << analytic_mesh
              << " nmm " << nmm
              ;

    std::string instance_slice = m_fcfg->getISlice() ;;
    std::string face_slice = m_fcfg->getFSlice() ;;
    std::string part_slice = m_fcfg->getPSlice() ;;

    NSlice* islice = !instance_slice.empty() ? new NSlice(instance_slice.c_str()) : NULL ; 
    NSlice* fslice = !face_slice.empty() ? new NSlice(face_slice.c_str()) : NULL ; 
    NSlice* pslice = !part_slice.empty() ? new NSlice(part_slice.c_str()) : NULL ; 

    for(int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_ggeo->getMergedMesh(i);
        if(restrict_mesh > -1 && i != restrict_mesh ) mm->setGeoCode(OpticksConst::GEOCODE_SKIP);      
        if(analytic_mesh > -1 && i == analytic_mesh && i > 0) 
        {
            GPmt* pmt = m_ggeo->getPmt(); 
            assert(pmt && "analyticmesh requires PMT resource");

            GParts* analytic = pmt->getParts() ;
            // TODO: the strings should come from config, as detector specific

            analytic->setVerbose(true); 
            analytic->setContainingMaterial("MineralOil");       
            analytic->setSensorSurface("lvPmtHemiCathodeSensorSurface");

            mm->setGeoCode(OpticksConst::GEOCODE_ANALYTIC);      
            mm->setParts(analytic);  
        }
        if(i>0) mm->setInstanceSlice(islice);

        // restrict to non-global for now
        if(i>0) mm->setFaceSlice(fslice);   
        if(i>0) mm->setPartSlice(pslice);   
    }

    TIMER("configureGeometry"); 
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

    m_opticks->setSpaceDomain( glm::vec4(ce0.x,ce0.y,ce0.z,ce0.w) );


    LOG(debug) << "OpticksGeometry::registerGeometry setting opticks SpaceDomain : " 
                      << " x " << ce0.x
                      << " y " << ce0.y
                      << " z " << ce0.z
                      << " w " << ce0.w
                      ;
 
    TIMER("registerGeometry"); 
}




