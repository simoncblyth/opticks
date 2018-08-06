#include "CFG4_BODY.hh"
// op --cgdmldetector
// op --ctestdetector

#include <cstdio>

// g4-
#include "G4PVPlacement.hh"
#include "G4GDMLParser.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalBorderSurface.hh"


// brap-
#include "BFile.hh"

// npy-
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "NPY.hpp"
#include "NBoundingBox.hpp"

// okc-
#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksResource.hh"
#include "OpticksQuery.hh"

#include "GGeo.hh"

// cfg4-
#include "CSurfaceLib.hh"
#include "CBndLib.hh"
#include "CMaterialLib.hh"
#include "CTraverser.hh"
#include "CDetector.hh"
#include "CCheck.hh"

#include "PLOG.hh"

CDetector::CDetector(OpticksHub* hub, OpticksQuery* query)
    : 
    m_hub(hub),
    m_ok(m_hub->getOpticks()),
    m_dbgsurf(m_ok->isDbgSurf()),
    m_ggb(m_hub->getGGeoBase()),
    m_blib(new CBndLib(m_hub)),
    m_query(query),
    m_resource(m_ok->getResource()),
    m_gmateriallib(m_hub->getMaterialLib()),
    m_mlib(new CMaterialLib(m_hub)),
    m_gsurfacelib(m_hub->getSurfaceLib()),
    m_slib(new CSurfaceLib(m_gsurfacelib)),   // << WIP 
    m_top(NULL),
    m_traverser(NULL),
    m_check(NULL),
    m_bbox(new NBoundingBox),
    m_verbosity(0),
    m_valid(true),
    m_level(info)
{
    init();
}


void CDetector::init()
{
    LOG(m_level) << "." ; 
}

bool CDetector::isValid()
{
    return m_valid ; 
}
void CDetector::setValid(bool valid)
{
    m_valid = valid ; 
}

void CDetector::setTop(G4VPhysicalVolume* top)
{
    LOG(m_level) << "." ; 
    m_top = top ; 
    traverse(m_top);
}

void CDetector::traverse(G4VPhysicalVolume* /*top*/)
{
    // invoked from CGDMLDetector::init OR CTestDetector::init via CDetector::setTop

    if(m_dbgsurf)
         LOG(info) << "[--dbgsurf] CDetector::traverse START " ;

    m_check = new CCheck(m_ok, m_top );
    
    m_traverser = new CTraverser(m_ok, m_top, m_bbox, m_query ); 
    m_traverser->Traverse();
    m_traverser->Summary("CDetector::traverse");

    if(m_dbgsurf)
         LOG(info) << "[--dbgsurf] CDetector::traverse DONE " ;
}

G4VPhysicalVolume* CDetector::Construct()
{
    return m_top ; 
}
G4VPhysicalVolume* CDetector::getTop()
{
    return m_top ; 
}
CMaterialLib* CDetector::getMaterialLib() const 
{
    return m_mlib ; 
}
GMaterialLib* CDetector::getGMaterialLib() const 
{
    return m_gmateriallib ; 
}

CSurfaceLib* CDetector::getSurfaceLib() const 
{
    return m_slib ; 
}
GSurfaceLib* CDetector::getGSurfaceLib() const 
{
    return m_gsurfacelib ; 
}


NBoundingBox* CDetector::getBoundingBox()
{
    return m_bbox ; 
}
void CDetector::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}



void CDetector::dumpLV(const char* msg)
{ 
    assert(m_traverser);
    m_traverser->dumpLV(msg);
}


const glm::vec4& CDetector::getCenterExtent()
{
    return m_bbox->getCenterExtent() ; 
}

void CDetector::saveBuffers(const char* objname, unsigned int objindex)
{
    assert(m_traverser);

    std::string cachedir = m_ok->getObjectPath(objname, objindex);

    NPY<float>* gtransforms = m_traverser->getGlobalTransforms(); 
    NPY<float>* ltransforms = m_traverser->getLocalTransforms(); 
    NPY<float>* center_extent = m_traverser->getCenterExtent(); 

    gtransforms->save(cachedir.c_str(), "gtransforms.npy");
    ltransforms->save(cachedir.c_str(), "ltransforms.npy");
    center_extent->save(cachedir.c_str(), "center_extent.npy");
}

unsigned int CDetector::getNumGlobalTransforms()
{
    assert(m_traverser);
    return m_traverser->getNumGlobalTransforms();
}
unsigned int CDetector::getNumLocalTransforms()
{
    assert(m_traverser);
    return m_traverser->getNumLocalTransforms();
}

glm::mat4 CDetector::getGlobalTransform(unsigned int index)
{
    assert(m_traverser);
    return m_traverser->getGlobalTransform(index);
}
glm::mat4 CDetector::getLocalTransform(unsigned int index)
{
    assert(m_traverser);
    return m_traverser->getLocalTransform(index);
}

NPY<float>* CDetector::getGlobalTransforms()
{
    assert(m_traverser);
    return m_traverser->getGlobalTransforms();
}

NPY<float>* CDetector::getLocalTransforms()
{
    assert(m_traverser);
    return m_traverser->getLocalTransforms();
}

const char* CDetector::getPVName(unsigned int index)
{
    assert(m_traverser);
    return m_traverser->getPVName(index);
}

const G4VPhysicalVolume* CDetector::getPV(unsigned index)
{
   return m_traverser->getPV(index); 
}
const G4VPhysicalVolume* CDetector::getPV(const char* name)
{
   return m_traverser->getPV(name); 
}

const G4LogicalVolume* CDetector::getLV(unsigned index)
{
   return m_traverser->getLV(index); 
}
const G4LogicalVolume* CDetector::getLV(const char* name)
{
   return m_traverser->getLV(name); 
}

//////// TODO get rid of m_pvm based methods, that rely on 
///////       manually setting m_pvm in CTestDetector

void CDetector::dumpLocalPV(const char* msg)
{
    LOG(info) << msg ; 

    typedef std::map<std::string, G4VPhysicalVolume*> MSV ; 

    for(MSV::const_iterator it=m_pvm.begin() ; it != m_pvm.end() ; it++)
    {
         std::string pvn = it->first ; 
         G4VPhysicalVolume* pv = it->second ;  

         std::cout << std::setw(40) << pvn 
                   << std::setw(40) << pv->GetName() 
                   << std::endl 
                   ;

    }
}

  
G4VPhysicalVolume* CDetector::getLocalPV(const char* name)
{
    return m_pvm.count(name) == 1 ? m_pvm[name] : NULL ; 
}

CDetector::~CDetector()
{
    //printf("CDetector::~CDetector\n");
    //G4GeometryManager::GetInstance()->OpenGeometry();
    //printf("CDetector::~CDetector DONE\n");
}



/**
CDetector::attachSurfaces
----------------------------

Older versions of GDML entirely omit the surfaces,  
so try to reconstruct them by conversion with CSurfaceLib 
from the Opticks surfaces held in GGeo/GSurfaceLib 

Formerly:

     Invoked from CGeometry::init immediately after 
     CTestDetector or GDMLDetector instanciation


Now done internally within CTestDetector::init and GDMLDetector::init
to avoid forgetting to do them.


Note that the conversion from the Opticks model back to the Geant4 one
excludes sensors : this is because they are "fakes" added to bring 
the Opticks surface model in line with the Geant4 volume model.

Geant4
    hit formed when photon track steps into sensitive volume
Opticks
    hit formed when photon reaches a sensitive surface 


Have observed that this adds EFFICIENCY and REFLECTIVITY 
properties to all surfaces, due to Opticks standardization
in GSurfaceLib.

**/

void CDetector::attachSurfaces()
{
    LOG(m_level) << "." ; 

    int num_bs = G4LogicalBorderSurface::GetNumberOfBorderSurfaces();
    int num_sk = G4LogicalSkinSurface::GetNumberOfSkinSurfaces();

    LOG(info) 
         << " num_bs " << num_bs
         << " num_sk " << num_sk
         ;

    if(num_bs > 0 || num_sk > 0)
    {
        LOG(error) << " some surfaces were found : so assume there is nothing to do " ; 
        return ; 
    }

    //if(m_dbgsurf)
        LOG(info) << "[--dbgsurf] CDetector::attachSurfaces START" ;

    bool exclude_sensors = true ; 
    m_slib->convert(this, exclude_sensors );

    //if(m_dbgsurf)
        LOG(info) << "[--dbgsurf] CDetector::attachSurfaces DONE " ;

} 


void CDetector::export_dae(const char* dir, const char* name)
{
    std::string path_ = BFile::FormPath(dir, name);

    const G4String path = path_ ; 
    LOG(info) << "export to " << path_ ; 

    G4VPhysicalVolume* world_pv = getTop();
    assert( world_pv  );

#ifdef WITH_G4DAE 
    G4DAEParser* g4dae = new G4DAEParser ;

    G4bool refs = true ;
    G4bool recreatePoly = false ; 
    G4int nodeIndex = -1 ;   // so World is volume 0 

    g4dae->Write(path, world_pv, refs, recreatePoly, nodeIndex );
#else
    LOG(warning) << " export requires WITH_G4DAE " ; 
#endif
}

void CDetector::export_gdml(const char* dir, const char* name)
{
    std::string path_ = BFile::FormPath(dir, name);
 
    m_check->checkSurf();
 
    const G4String path = path_ ; 
    LOG(info) << "export to " << path_ ; 

    G4VPhysicalVolume* world_pv = getTop();
    assert( world_pv  );

    G4GDMLParser* g4gdml = new G4GDMLParser ;
    G4bool refs = true ;
    G4String schemaLocation = "" ; 

    g4gdml->Write(path, world_pv, refs, schemaLocation );

}


