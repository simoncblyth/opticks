#include <cassert>
#include <cstdio>
#include <cstring>
#include <iomanip>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "SLog.hh"
#include "BStr.hh"
#include "BMap.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NQuad.hpp"
#include "NMeta.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "TorchStepNPY.hpp"

#include "NSensor.hpp"

#include "NLookup.hpp"
#include "NSlice.hpp"
#include "Typ.hpp"


// opticks-
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksColors.hh"
#include "OpticksFlags.hh"
#include "OpticksAttrSeq.hh"
#include "Composition.hh"

// ggeo-

#include "GDomain.hh"
#include "GAry.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GPropertyLib.hh"
#include "GOpticalSurface.hh"
#include "GSkinSurface.hh"
#include "GBorderSurface.hh"
#include "GMaterial.hh"

#include "GMeshLib.hh"
#include "GNodeLib.hh"
#include "GGeoLib.hh"
#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GScintillatorLib.hh"
#include "GSourceLib.hh"


#include "GVolume.hh"
#include "GMesh.hh"
#include "GInstancer.hh"
#include "GTreePresent.hh"
#include "GColorizer.hh"
#include "GPmtLib.hh"
#include "GScene.hh"

#include "GMergedMesh.hh"
#include "GItemIndex.hh"
#include "GItemList.hh"

#include "GGeo.hh"

#include "GGEO_BODY.hh"
#include "PLOG.hh"

#define BSIZ 50


const plog::Severity GGeo::LEVEL = debug ; 


const char* GGeo::CATHODE_MATERIAL = "Bialkali" ; 
const char* GGeo::PICKFACE = "pickface" ;

GGeo* GGeo::fInstance = NULL ; 

GGeo* GGeo::GetInstance()
{
    return fInstance ;    
}

GGeo::GGeo(Opticks* ok)
  :
   m_log(new SLog("GGeo::GGeo","",verbose)),
   m_ok(ok), 
   m_analytic(false),
   m_gltf(m_ok->getGLTF()),   
   m_composition(NULL), 
   m_instancer(NULL), 
   m_loaded(false), 
   m_prepared(false), 
   m_cachemeta(NULL),
   m_lv2sd(NULL),
   m_lookup(NULL), 
   m_meshlib(NULL),
   m_geolib(NULL),
   m_nodelib(NULL),
   m_bndlib(NULL),
   m_materiallib(NULL),
   m_surfacelib(NULL),
   m_scintillatorlib(NULL),
   m_sourcelib(NULL),
   m_pmtlib(NULL),
   m_colorizer(NULL),
   m_low(NULL),
   m_high(NULL),
   m_sensitive_count(0),
   m_join_cfg(NULL),
   m_mesh_verbosity(0),
   m_gscene(NULL)
{
   init(); 
   (*m_log)("DONE"); 

   if(fInstance != NULL) LOG(error) << " replacing GGeo::fInstance " ; 
   fInstance = this ; 
}



// setLoaderImp : sets implementation that does the actual loading
// using a function pointer to the implementation 
// avoids ggeo-/GLoader depending on all the implementations

void GGeo::setLoaderImp(GLoaderImpFunctionPtr imp)
{
    m_loader_imp = imp ; 
}

void GGeo::setComposition(Composition* composition)
{
    m_composition = composition ; 
}
Composition* GGeo::getComposition()
{
    return m_composition ; 
}

void GGeo::setMeshVerbosity(unsigned int verbosity)
{
    m_mesh_verbosity = verbosity  ; 
}
unsigned int GGeo::getMeshVerbosity() const 
{
    return m_mesh_verbosity ;
}


void GGeo::setMeshJoinImp(GJoinImpFunctionPtr imp)
{
    m_join_imp = imp ; 
}
void GGeo::setMeshJoinCfg(const char* cfg)
{
    m_join_cfg = cfg ? strdup(cfg) : NULL  ; 
}

bool GGeo::isLoaded()
{
    return m_loaded ; 
}





///////// from m_surfacelib ////////////

void GGeo::add(GBorderSurface* surface)
{
    m_surfacelib->add(surface);
}
void GGeo::add(GSkinSurface* surface)
{
    m_surfacelib->add(surface);
}
unsigned int GGeo::getNumBorderSurfaces() const 
{
    return m_surfacelib->getNumBorderSurfaces() ; 
}
unsigned int GGeo::getNumSkinSurfaces() const 
{
    return m_surfacelib->getNumSkinSurfaces() ; 
}
GSkinSurface* GGeo::getSkinSurface(unsigned index) const 
{
    return m_surfacelib->getSkinSurface(index); 
}
GBorderSurface* GGeo::getBorderSurface(unsigned index) const 
{
    return m_surfacelib->getBorderSurface(index); 
}
void GGeo::addRaw(GBorderSurface* surface)
{
    m_surfacelib->addRaw(surface);
}
void GGeo::addRaw(GSkinSurface* surface)
{
    m_surfacelib->addRaw(surface);
}
unsigned GGeo::getNumRawBorderSurfaces() const 
{
    return m_surfacelib->getNumRawBorderSurfaces() ; 
}
unsigned GGeo::getNumRawSkinSurfaces() const 
{
    return m_surfacelib->getNumRawSkinSurfaces() ;
}




/// GGeoBase interface ////

GScintillatorLib* GGeo::getScintillatorLib() const { return m_scintillatorlib ; }
GSourceLib*       GGeo::getSourceLib() const  { return m_sourcelib ; }
GSurfaceLib*      GGeo::getSurfaceLib() const { return m_surfacelib ; } 
GMaterialLib*     GGeo::getMaterialLib() const { return m_materiallib ; }
 
GBndLib*          GGeo::getBndLib() const { return m_bndlib ; } 
GPmtLib*          GGeo::getPmtLib() const { return m_pmtlib ; } 
GGeoLib*          GGeo::getGeoLib() const { return m_geolib ; } 
GNodeLib*         GGeo::getNodeLib() const { return m_nodelib ; } 

const char*       GGeo::getIdentifier() const  { return "GGeo" ; } 

/// END GGeoBase interface ////


void GGeo::setLookup(NLookup* lookup)
{
    m_lookup = lookup ; 
}


NLookup* GGeo::getLookup()
{
    return m_lookup ; 
}

GColorizer* GGeo::getColorizer()
{
    return m_colorizer ; 
}


gfloat3* GGeo::getLow()
{
   return m_low ; 
}
gfloat3* GGeo::getHigh()
{
   return m_high ; 
}


GInstancer* GGeo::getTreeCheck()
{
    return m_instancer ;
}




/**
GGeo::setCathode
------------------

Invoked from AssimpGGeo::convertMaterials

**/
void GGeo::setCathode(GMaterial* cathode)
{
    m_materiallib->setCathode(cathode); 
}
GMaterial* GGeo::getCathode() const 
{
    return m_materiallib->getCathode() ; 
}
const char* GGeo::getCathodeMaterialName() const
{
    return m_materiallib->getCathodeMaterialName() ; 
}



/**
GGeo::addLVSD
-------------------

From  

1. AssimpGGeo::convertSensorsVisit
2. X4PhysicalVolume::convertSensors_r

**/

void GGeo::addLVSD(const char* lv, const char* sd)
{
   assert( lv ) ;  
   m_cathode_lv.insert(lv);

   if(sd) 
   {
       if(m_lv2sd == NULL ) m_lv2sd = new NMeta ; 
       m_lv2sd->set<std::string>(lv, sd) ; 
   }
}
unsigned GGeo::getNumLVSD() const
{
   return m_lv2sd ? m_lv2sd->getNumKeys() : 0 ;  
}
std::pair<std::string,std::string> GGeo::getLVSD(unsigned idx) const
{
    const char* lv = m_lv2sd->getKey(idx) ; 
    std::string sd = m_lv2sd->get<std::string>(lv); 
    return std::pair<std::string,std::string>( lv, sd ); 
}





int GGeo::findCathodeLVIndex(const char* lv) const  // -1 if not found
{
    int index = -1 ; 
    if( lv == NULL ) return index ; 


    typedef std::unordered_set<std::string>::const_iterator UCI ; 
    UCI b = m_cathode_lv.begin() ;
    UCI e = m_cathode_lv.end() ;

    for(UCI it=b ; it != e ; it++)
    {
        const char* clv = it->c_str(); 
        if( strcmp( clv, lv) == 0)
        {
            index = std::distance( b, it ) ; 
            break ; 
        }
    }

    if( index > -1 )
    {
        const char* clv2 = getCathodeLV(index); 
        assert( strcmp(lv, clv2) == 0 ) ;  
    }

    return index ; 
}

unsigned int GGeo::getNumCathodeLV() const 
{
   return m_cathode_lv.size() ; 
}
const char* GGeo::getCathodeLV(unsigned int index) const 
{
    typedef std::unordered_set<std::string>::const_iterator UCI ; 
    UCI it = m_cathode_lv.begin() ; 
    std::advance( it, index );
    return it != m_cathode_lv.end() ? it->c_str() : NULL  ; 
}

void GGeo::dumpCathodeLV(const char* msg) const 
{
    //printf("%s\n", msg);
    LOG(LEVEL) << msg ; 

    typedef std::unordered_set<std::string>::const_iterator UCI ; 
    for(UCI it=m_cathode_lv.begin() ; it != m_cathode_lv.end() ; it++)
    {
        //printf("GGeo::dumpCathodeLV %s \n", it->c_str() ); 
        LOG(LEVEL) << it->c_str() ;  
    }
}
 
void GGeo::getCathodeLV( std::vector<std::string>& lvnames ) const 
{
    typedef std::unordered_set<std::string>::const_iterator UCI ; 
    for(UCI it=m_cathode_lv.begin() ; it != m_cathode_lv.end() ; it++) 
         lvnames.push_back(*it) ; 
}





Opticks* GGeo::getOpticks() const 
{
    return m_ok ; 
}








void GGeo::init()
{
   const char* idpath = m_ok->getIdPath() ;
   LOG(verbose)
         << " idpath " << ( idpath ? idpath : "NULL" )
         ; 

   assert(idpath && "GGeo::init idpath is required" );

   fs::path geocache(idpath); 
   bool cache_exists = fs::exists(geocache) && fs::is_directory(geocache) ;
   bool cache_requested = m_ok->isGeocache() ; 

   m_loaded = cache_exists && cache_requested ;

   LOG(LEVEL) 
        << " idpath " << idpath
        << " cache_exists " << cache_exists 
        << " cache_requested " << cache_requested
        << " m_loaded " << m_loaded 
        ;

   if(m_loaded) return ; 

   //////////////  below only when operating pre-cache //////////////////////////

   m_bndlib = new GBndLib(m_ok);
   m_materiallib = new GMaterialLib(m_ok);
   m_surfacelib  = new GSurfaceLib(m_ok);

   m_bndlib->setMaterialLib(m_materiallib);
   m_bndlib->setSurfaceLib(m_surfacelib);

   // NB this m_analytic is always false
   //    the analytic versions of these libs are born in GScene
   assert( m_analytic == false );  
   bool testgeo = false ;  

   m_meshlib = new GMeshLib(m_ok, m_analytic);
   m_geolib = new GGeoLib(m_ok, m_analytic, m_bndlib );
   m_nodelib = new GNodeLib(m_ok, m_analytic, testgeo ); 

   m_instancer = new GInstancer(m_geolib, m_nodelib, m_ok->getSceneConfig() ) ;


   GColorizer::Style_t style = GColorizer::PSYCHEDELIC_NODE ;
   OpticksColors* colors = getColors();

   m_colorizer = new GColorizer( m_nodelib, m_geolib, m_bndlib, colors, style ); // colorizer needs full tree, so pre-cache only 


   m_scintillatorlib  = new GScintillatorLib(m_ok);
   m_sourcelib  = new GSourceLib(m_ok);

   m_pmtlib = NULL ; 

   LOG(verbose) << "GGeo::init DONE" ; 
}


void GGeo::add(GMaterial* material)
{

    m_materiallib->add(material);
    //addToIndex((GPropertyMap<float>*)material);
}
void GGeo::addRaw(GMaterial* material)
{
    m_materiallib->addRaw(material);
}




unsigned GGeo::getNumMaterials() const 
{
    return m_materiallib->getNumMaterials();
}
unsigned GGeo::getNumRawMaterials() const 
{
    return m_materiallib->getNumRawMaterials();
}

GScene* GGeo::getScene() 
{
    return m_gscene ; 
}


unsigned int GGeo::getNumMergedMesh()
{
    GGeoLib* geolib = getGeoLib() ;
    assert(geolib);
    return geolib->getNumMergedMesh();
}

GMergedMesh* GGeo::getMergedMesh(unsigned index) const 
{
    GGeoLib* geolib = getGeoLib() ;
    assert(geolib);

    GMergedMesh* mm = geolib->getMergedMesh(index);

    unsigned meshverbosity = getMeshVerbosity() ; 

    LOG(debug) << "GGeo::getMergedMesh"
              << " index " << index 
              << " mm " << mm
              << " meshverbosity " << meshverbosity
              ;

    if(mm)
        mm->setVerbosity(meshverbosity);

    return mm ; 
}


const char* GGeo::getIdPath()
{
    return m_ok->getIdPath();
}
OpticksColors* GGeo::getColors()
{
   return m_ok->getColors() ; 
}
OpticksFlags* GGeo::getFlags()
{
    return m_ok->getFlags();
}
OpticksAttrSeq* GGeo::getFlagNames()
{
    return m_ok->getFlagNames();
} 

OpticksResource* GGeo::getResource()
{
    return m_ok->getResource();
}






unsigned int GGeo::getMaterialLine(const char* shortname)
{
    return m_bndlib->getMaterialLine(shortname);
}

void GGeo::loadGeometry()
{
    bool loaded = isLoaded() ;

    int gltf = m_ok->getGLTF(); 

    LOG(info) << "GGeo::loadGeometry START" 
              << " loaded " << loaded 
              << " gltf " << gltf
              ; 

    if(!loaded)
    {
        loadFromG4DAE();
        save();

        if(gltf > 0 && gltf < 10) 
        {
            loadAnalyticFromGLTF();
            saveAnalytic();
        }
    }
    else
    {
        loadFromCache();
        if(gltf > 0 && gltf < 10)  
        {
            loadAnalyticFromCache(); 
        }
    } 


    if(m_ok->isAnalyticPMTLoad())
    {
        m_pmtlib = GPmtLib::load(m_ok, m_bndlib );
    }

    if( gltf >= 10 )
    {
        LOG(info) << "GGeo::loadGeometry DEBUGGING loadAnalyticFromGLTF " ; 
        loadAnalyticFromGLTF();
    }


    // HMM : this not done in direct route ?
    setupLookup();
    setupColors();
    setupTyp();

    LOG(info) << "GGeo::loadGeometry DONE" ; 
}

void GGeo::loadFromG4DAE()
{
    LOG(error) << "GGeo::loadFromG4DAE START" ; 

    int rc = (*m_loader_imp)(this);   //  imp set in OpticksGeometry::loadGeometryBase, m_ggeo->setLoaderImp(&AssimpGGeo::load); 

    if(rc != 0)
        LOG(fatal) << "GGeo::loadFromG4DAE"
                   << " FAILED : probably you need to download opticksdata "
                   ;

    assert(rc == 0 && "G4DAE geometry file does not exist, try : opticksdata- ; opticksdata-- ") ;

    prepare();

    LOG(error) << "GGeo::loadFromG4DAE DONE" ; 
}



/**
GGeo::postDirectTranslation
-------------------------------

Invoked from G4Opticks::translateGeometry after the X4PhysicalVolume conversion

**/


void GGeo::postDirectTranslation()
{
    LOG(debug) << "[" ; 

    LOG(info) << "( GGeo::prepare " ; 
    prepare();         
    LOG(info) << ") GGeo::prepare " ; 

    LOG(info) << "( GBndLib::fillMaterialLineMap " ; 
    GBndLib* blib = getBndLib();
    blib->fillMaterialLineMap();
    LOG(info) << ") GBndLib::fillMaterialLineMap " ; 

    LOG(info) << "( GGeo::save " ; 
    save();
    LOG(info) << ") GGeo::save " ; 

    LOG(debug) << "]" ; 
}

bool GGeo::isPrepared() const { return m_prepared ; }
void GGeo::prepare()
{
   // prepare is needed prior to saving or GPU upload by OGeo
    assert( m_prepared == false && "have prepared already" ); 
    m_prepared = true ; 

    //TODO: implement prepareSensorSurfaces() and invoke from here 

   
    LOG(info) << "prepareScintillatorLib" ;  
    prepareScintillatorLib();

    LOG(info) << "prepareSourceLib" ;  
    prepareSourceLib();

    LOG(info) << "prepareVolumes" ;  
    prepareVolumes();   // GInstancer::createInstancedMergedMeshes

    LOG(info) << "prepareVertexColors" ;  
    prepareVertexColors();  // writes colors into GMergedMesh mm0
}


void GGeo::loadAnalyticFromGLTF()
{
    LOG(info) << "GGeo::loadAnalyticFromGLTF START" ; 
    if(!m_ok->isGLTF()) return ; 

#ifdef OPTICKS_YoctoGL
    m_gscene = GScene::Create(m_ok, this); 
#else
    LOG(fatal) << "GGeo::loadAnalyticFromGLTF requires YoctoGL external " ; 
    assert(0);
#endif

    LOG(info) << "GGeo::loadAnalyticFromGLTF DONE" ; 
}




void GGeo::save()
{
    const char* idpath = m_ok->getIdPath() ;
    LOG(LEVEL) << "[" 
              << " idpath " << ( idpath ? idpath : "NULL" )
               ;


    if(!m_prepared)
    {
        LOG(info) << "preparing before save " ; 
        prepare();
    }   


    m_geolib->dump("GGeo::save");

    m_geolib->save();
    m_meshlib->save();
    m_nodelib->save();
    m_materiallib->save();
    m_surfacelib->save();
    m_scintillatorlib->save();
    m_sourcelib->save();
    m_bndlib->save();  

    saveCacheMeta();

    LOG(LEVEL) << "]" ;  
}


void GGeo::saveCacheMeta()
{
    if(m_cachemeta == NULL ) m_cachemeta = new NMeta ; 

    m_cachemeta->set<int>("answer", 42) ; 
    m_cachemeta->set<std::string>("question", "huh?");

    //LOG(error) << " PLOG::instance " << PLOG::instance ; 

    std::string argline = PLOG::instance->args.argline() ; 
    m_cachemeta->set<std::string>("argline", argline ); 

    if(m_lv2sd) m_cachemeta->setObj("lv2sd", m_lv2sd ); 
    const char* path = m_ok->getCacheMetaPath(); 
    m_cachemeta->save(path); 
}
void GGeo::loadCacheMeta()
{
    const char* path = m_ok->getCacheMetaPath(); 
    assert( m_cachemeta == NULL ); 
    m_cachemeta = NMeta::Load(path);
    m_lv2sd = m_cachemeta->getObj("lv2sd"); 
}


void GGeo::saveAnalytic()
{ 
    LOG(info) << "GGeo::saveAnalytic" ;
    m_gscene->save();   // HUH: still needed ???
}
 
void GGeo::loadFromCache()
{   
    LOG(error) << "GGeo::loadFromCache START" ; 

    bool constituents = true ; 
    m_bndlib = GBndLib::load(m_ok, constituents);    // interpolation potentially happens in here

    // GBndLib is persisted via index buffer, not float buffer
    m_materiallib = m_bndlib->getMaterialLib();
    m_surfacelib = m_bndlib->getSurfaceLib();

    m_scintillatorlib  = GScintillatorLib::load(m_ok);
    m_sourcelib  = GSourceLib::load(m_ok);

    bool analytic = false ; 
    bool testgeo = false ; 

    m_geolib = GGeoLib::Load(m_ok, analytic, m_bndlib);
    m_nodelib = GNodeLib::Load(m_ok, analytic, testgeo );        
    m_meshlib = GMeshLib::Load(m_ok, analytic);

    loadCacheMeta();

    LOG(error) << "GGeo::loadFromCache DONE" ; 
}

void GGeo::loadAnalyticFromCache()
{
    LOG(info) << "GGeo::loadAnalyticFromCache START" ; 
    m_gscene = GScene::Load(m_ok, this); // GGeo needed for m_bndlib 
    LOG(info) << "GGeo::loadAnalyticFromCache DONE" ; 
}


bool GGeo::isValid()
{
    return m_bndlib->isValid() && m_materiallib->isValid() && m_surfacelib->isValid() ; 
}


void GGeo::afterConvertMaterials()
{
    LOG(debug) << "GGeo::afterConvertMaterials and before convertStructure" ; 

    prepareMaterialLib(); 
    prepareSurfaceLib(); 
}



void GGeo::setupLookup()
{
    assert(m_lookup && "must GGeo::setLookup before can load geometry, normally done by OpticksGeometry::init " );

    // setting of lookup A, now moved up to OpticksHub::configureLookup

    const std::map<std::string, unsigned int>& B  = m_bndlib->getMaterialLineMap();

    m_lookup->setB(B,"", "GGeo::setupLookup/m_bndlib") ;
}

void GGeo::setupTyp()
{
   // hmm maybe better elsewhere to avoid repetition from tests ? 
    Typ* typ = m_ok->getTyp();
    typ->setMaterialNames(m_materiallib->getNamesMap());
    typ->setFlagNames(m_ok->getFlagNamesMap());
}

void GGeo::setupColors()
{
    LOG(verbose) << "GGeo::setupColors" ; 

    //OpticksFlags* flags = m_ok->getFlags();

    std::vector<unsigned int>& material_codes = m_materiallib->getAttrNames()->getColorCodes() ; 
    std::vector<unsigned int>& flag_codes     = m_ok->getFlagNames()->getColorCodes() ; 

    OpticksColors* colors = m_ok->getColors();

    colors->setupCompositeColorBuffer( material_codes, flag_codes  );

    LOG(verbose) << "GGeo::setupColors DONE" ; 
}


void GGeo::setLow(const gfloat3& low)
{
    m_low = new gfloat3(low);
}
void GGeo::setHigh(const gfloat3& high)
{
    m_high = new gfloat3(high);
}

void GGeo::updateBounds(GNode* node)
{
    if(!m_low)  m_low  = new gfloat3(1e10f, 1e10f, 1e10f) ;
    if(!m_high) m_high = new gfloat3(-1e10f, -1e10f, -1e10f) ;
  
    node->updateBounds(*m_low, *m_high);
}

void GGeo::Summary(const char* msg)
{
    LOG(info) << msg
              << " ms " << m_meshlib->getNumMeshes()
              << " so " << m_nodelib->getNumVolumes()
              << " mt " << m_materiallib->getNumMaterials()
              << " bs " << getNumBorderSurfaces() 
              << " ss " << getNumSkinSurfaces()
              ;

    if(m_low)  printf("    low  %10.3f %10.3f %10.3f \n", m_low->x, m_low->y, m_low->z);
    if(m_high) printf("    high %10.3f %10.3f %10.3f \n", m_high->x, m_high->y, m_high->z);
}

void GGeo::Details(const char* msg)
{
    Summary(msg) ;

    char mbuf[BSIZ];

    /*
    for(unsigned int ims=0 ; ims < m_meshes.size()  ; ims++ )
    {
        GMesh* ms = m_meshes[ims];
        snprintf(mbuf,BSIZ, "%s ms %u", msg, ims);
        ms->Summary(mbuf);
    }
    */

    for(unsigned int ibs=0 ; ibs < getNumBorderSurfaces()  ; ibs++ )
    {
        GBorderSurface* bs = getBorderSurface(ibs);
        snprintf(mbuf,BSIZ, "%s bs %u", msg, ibs);
        bs->Summary(mbuf);
    }
    for(unsigned int iss=0 ; iss < getNumSkinSurfaces()  ; iss++ )
    {
        GSkinSurface* ss = getSkinSurface(iss) ;
        snprintf(mbuf,BSIZ, "%s ss %u", msg, iss);
        ss->Summary(mbuf);
    }
    for(unsigned int imat=0 ; imat < m_materiallib->getNumMaterials()  ; imat++ )
    {
        GMaterial* mat = m_materiallib->getMaterial(imat) ;
        snprintf(mbuf,BSIZ, "%s mt %u", msg, imat);
        mat->Summary(mbuf);
    }

    /*
    for(unsigned int isol=0 ; isol < m_volumes.size()  ; isol++ )
    {
        GVolume* sol = m_volumes[isol];
        snprintf(mbuf,BSIZ, "%s so %u", msg, isol);
        sol->Summary(mbuf);
    }
    */
}




//  via meshlib

GMeshLib* GGeo::getMeshLib()
{
    return m_meshlib ; 
}
unsigned GGeo::getNumMeshes() const 
{
    return m_meshlib->getNumMeshes(); 
}
GItemIndex* GGeo::getMeshIndex()
{
    return m_meshlib->getMeshIndex() ; 
}
const GMesh* GGeo::getMesh(unsigned aindex) const 
{
    return m_meshlib->getMesh(aindex);
}  
void GGeo::add(const GMesh* mesh)
{
    //assert(0);
    m_meshlib->add(mesh);
}
void GGeo::countMeshUsage(unsigned meshIndex, unsigned nodeIndex)
{
    m_meshlib->countMeshUsage(meshIndex, nodeIndex); 
}
void GGeo::reportMeshUsage(const char* msg)
{
    m_meshlib->reportMeshUsage(msg);
}
 



// via GNodeLib

unsigned GGeo::getNumVolumes() const 
{
    return m_nodelib->getNumVolumes();
}
void GGeo::add(GVolume* volume)
{
    m_nodelib->add(volume);
}
GVolume* GGeo::getVolume(unsigned index) const 
{
    return m_nodelib->getVolume(index);
}
GVolume* GGeo::getVolumeSimple(unsigned int index) const 
{
    return m_nodelib->getVolumeSimple(index);
}
const char* GGeo::getPVName(unsigned int index) const 
{
    return m_nodelib->getPVName(index);
}
const char* GGeo::getLVName(unsigned int index) const 
{
    return m_nodelib->getLVName(index);
}
GNode* GGeo::getNode(unsigned index) const 
{
    return m_nodelib->getNode(index);
}








#if 0

// cannot do this check any more in GBoundary approach 

void GGeo::materialConsistencyCheck()
{
    GVolume* volume = getVolume(0);
    assert(volume);
    unsigned int nok = materialConsistencyCheck(volume);
    printf("GGeo::materialConsistencyCheck nok %u \n", nok );
}

unsigned int GGeo::materialConsistencyCheck(GVolume* volume)
{
    assert(volume);
    //volume->Summary(NULL);

    GVolume* parent = dynamic_cast<GVolume*>(volume->getParent()) ; 

    unsigned int nok = 0 ;
    if(parent)
    {
        assert(parent->getInnerMaterial() == volume->getOuterMaterial());
        nok += 1 ;
    } 
    else
    {
        assert(volume->getIndex() == 0); 
    } 

    for(unsigned int i=0 ; i < volume->getNumChildren() ; i++)
    {
        GVolume* child = dynamic_cast<GVolume*>(volume->getChild(i)) ;
        assert(child); 
        nok += materialConsistencyCheck(child);
    }
    return nok ;
}

#endif




GMaterial* GGeo::getMaterial(unsigned aindex) const 
{
    return m_materiallib->getMaterialWithIndex(aindex);
}
GPropertyMap<float>* GGeo::findRawMaterial(const char* shortname) const 
{
    return m_materiallib->findRawMaterial(shortname) ; 
}
GProperty<float>* GGeo::findRawMaterialProperty(const char* shortname, const char* propname) const 
{
   return m_materiallib->findRawMaterialProperty(shortname, propname) ;   
}
void GGeo::dumpRawMaterialProperties(const char* msg) const 
{
    m_materiallib->dumpRawMaterialProperties(msg);
}
std::vector<GMaterial*> GGeo::getRawMaterialsWithProperties(const char* props, char delim) const 
{
    return m_materiallib->getRawMaterialsWithProperties(props, delim ); 
}






GSkinSurface* GGeo::findSkinSurface(const char* lv) const
{
    return m_surfacelib->findSkinSurface(lv);
}
GBorderSurface* GGeo::findBorderSurface(const char* pv1, const char* pv2) const 
{
    return m_surfacelib->findBorderSurface(pv1, pv2); 
}


void GGeo::dumpSkinSurface(const char* name) const
{
    m_surfacelib->dumpSkinSurface(name); 
}
void GGeo::dumpRawSkinSurface(const char* name) const
{
    m_surfacelib->dumpRawSkinSurface(name); 
}

void GGeo::dumpRawBorderSurface(const char* name) const 
{
    m_surfacelib->dumpRawBorderSurface(name); 
}





void GGeo::traverse(const char* msg)
{
    LOG(info) << msg ; 
    traverse( getVolume(0), 0 );
}


void GGeo::traverse( GNode* node, unsigned int depth)
{
    GVolume* volume = dynamic_cast<GVolume*>(node) ;

    NSensor* sensor = volume->getSensor(); 

    if(sensor)
         LOG(debug) << "GGeo::traverse " 
                   << " nodeIndex " << node->getIndex()
                   << sensor->description() 
                   ; 

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1);
}





void GGeo::prepareMaterialLib()
{
    LOG(verbose) << "GGeo::prepareMaterialLib " ; 

    GMaterialLib* mlib = getMaterialLib() ;
   
    mlib->addTestMaterials(); 
}

void GGeo::prepareSurfaceLib()
{
    LOG(verbose) << "GGeo::prepareSurfaceLib " ; 

    GSurfaceLib* slib = getSurfaceLib() ;
   
    slib->addPerfectSurfaces(); 
}


void GGeo::prepareSourceLib()
{
    LOG(verbose) << "GGeo::prepareSourceLib " ; 

    GSourceLib* srclib = getSourceLib() ;

    srclib->close();
}




void GGeo::close()
{
    LOG(fatal) << "[" ; 
    // this needs to be invoked after all Opticks materials and surfaces have been
    // created, and before boundaries are formed : typically in the recursive structure traverse

    GMaterialLib* mlib = getMaterialLib() ;
    GSurfaceLib* slib = getSurfaceLib() ;

    mlib->close();
    slib->close();


    // this was not here traditionally due to late addition of boundaries 
    GBndLib* blib = getBndLib() ;
    blib->createDynamicBuffers(); 


    LOG(fatal) << "]" ; 
}


//////////////////////  TODO : MOVE SCINT HANDLING INTO THE LIB

void GGeo::prepareScintillatorLib()
{
    LOG(verbose) << "GGeo::prepareScintillatorLib " ; 

    findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB"); 

    unsigned int nscint = getNumScintillatorMaterials() ;

    if(nscint == 0)
    {
        LOG(LEVEL) << " found no scintillator materials  " ; 
    }
    else
    {
        LOG(LEVEL) << " found " << nscint << " scintillator materials  " ; 

        GScintillatorLib* sclib = getScintillatorLib() ;

        for(unsigned int i=0 ; i < nscint ; i++)
        {
            GPropertyMap<float>* scint = dynamic_cast<GPropertyMap<float>*>(getScintillatorMaterial(i));  
            sclib->add(scint);
        }

        sclib->close(); 
    }
}

void GGeo::findScintillatorMaterials(const char* props)
{
    m_scintillators_raw = getRawMaterialsWithProperties(props, ',');
    //assert(m_scintillators_raw.size() > 0 );
}

void GGeo::dumpScintillatorMaterials(const char* msg)
{
    LOG(info)<< msg ;
    for(unsigned int i=0; i<m_scintillators_raw.size() ; i++)
    {
        GMaterial* mat = m_scintillators_raw[i];
        //mat->Summary();
        std::cout << std::setw(30) << mat->getShortName()
                  << " keys: " << mat->getKeysString()
                  << std::endl ; 
    }              
}

unsigned int GGeo::getNumScintillatorMaterials()
{
    return m_scintillators_raw.size();
}

GMaterial* GGeo::getScintillatorMaterial(unsigned int index)
{
    return index < m_scintillators_raw.size() ? m_scintillators_raw[index] : NULL ; 
}


//////////////////////



/**
GGeo::prepareVolumes
--------------------

This was formerly mis-named as prepareMeshes which as it 
also does the analytic combination, with analytic GParts 
instances hitched to the created GMergedMesh.

**/

void GGeo::prepareVolumes()
{
    bool instanced = m_ok->isInstanced();
    unsigned meshverbosity = m_ok->getMeshVerbosity() ; 

    m_instancer->setCSGSkipLV(m_ok->getCSGSkipLV()) ;  


    LOG(LEVEL) 
               << "START" 
               << " instanced " << instanced 
               << " meshverbosity " << meshverbosity 
              ;

    if(instanced)
    { 
        bool deltacheck = true ; 
        m_instancer->createInstancedMergedMeshes(deltacheck, meshverbosity);   // GInstancer::createInstancedMergedMeshes
    }
    else
    {
        LOG(fatal) << "GGeo::prepareVolumes instancing inhibited " ;
        GNode* root = getNode(0);
        m_geolib->makeMergedMesh(0, NULL, root, meshverbosity);  // ridx:0 rbase:NULL 
        // ^^^^  precache never needs analytic geolib ?
    }


    m_instancer->dumpMeshset() ; 

    LOG(LEVEL) << "DONE" ;
}





GMergedMesh* GGeo::makeMergedMesh(unsigned int index, GNode* base, GNode* root, unsigned verbosity )
{
    GGeoLib* geolib = getGeoLib() ;
    assert(geolib);
    return geolib->makeMergedMesh(index, base, root, verbosity);
}





void GGeo::prepareVertexColors()
{
    // GColorizer needs full tree,  so have to use pre-cache
    LOG(verbose) << "GGeo::prepareVertexColors START" ;
    m_colorizer->writeVertexColors();
    LOG(verbose) << "GGeo::prepareVertexColors DONE " ;
}






void GGeo::findCathodeMaterials(const char* props)
{
    m_cathodes_raw = getRawMaterialsWithProperties(props, ',');
    assert(m_cathodes_raw.size() > 0 );
}
void GGeo::dumpCathodeMaterials(const char* msg)
{
    LOG(info)<< msg ;
    for(unsigned int i=0; i<m_cathodes_raw.size() ; i++)
    {
        GMaterial* mat = m_cathodes_raw[i];
        mat->Summary();
        std::cout << std::setw(30) << mat->getShortName()
                  << " keys: " << mat->getKeysString()
                  << std::endl ; 
    }              
}

unsigned int GGeo::getNumCathodeMaterials()
{
    return m_cathodes_raw.size();
}

GMaterial* GGeo::getCathodeMaterial(unsigned int index)
{
    return index < m_cathodes_raw.size() ? m_cathodes_raw[index] : NULL ; 
}
















void GGeo::dumpStats(const char* msg)
{
    printf("%s\n", msg);

    unsigned int nmm = getNumMergedMesh();

    unsigned int totVertices(0);
    unsigned int totFaces(0);
    unsigned int vtotVertices(0);
    unsigned int vtotFaces(0);

    for(unsigned int i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = getMergedMesh(i);
        if(!mm) continue ; 

        GBuffer* tbuf = mm->getTransformsBuffer();
        NPY<float>* ibuf = mm->getITransformsBuffer();
        GBuffer* vbuf = mm->getVerticesBuffer();
        GBuffer* fbuf = mm->getIndicesBuffer();

        unsigned int numVertices = vbuf->getNumItems() ;
        unsigned int numFaces = fbuf->getNumItems()/3;

        unsigned int numTransforms = tbuf ? tbuf->getNumItems() : 1  ;
        unsigned int numITransforms = ibuf ? ibuf->getNumItems() : 0  ;


        if( i == 0)
        {
            totVertices += numVertices ; 
            totFaces    += numFaces ; 
            vtotVertices += numVertices ; 
            vtotFaces    += numFaces  ;
        }
        else
        {
            totVertices += numVertices ; 
            totFaces    += numFaces ; 
            vtotVertices += numVertices*numITransforms ; 
            vtotFaces    += numFaces*numITransforms  ;
        }

        printf(" mm %2d : vertices %7d faces %7d transforms %7d itransforms %7d \n", i, numVertices, numFaces, numTransforms, numITransforms );
 
    } 

    printf("   totVertices %9d  totFaces %9d \n", totVertices, totFaces );
    printf("  vtotVertices %9d vtotFaces %9d (virtual: scaling by transforms)\n", vtotVertices, vtotFaces );
    printf("  vfacVertices %9.3f vfacFaces %9.3f (virtual to total ratio)\n", float(vtotVertices)/float(totVertices), float(vtotFaces)/float(totFaces) );
}


void GGeo::dumpNodeInfo(unsigned int mmindex, const char* msg)
{
    GMergedMesh* mm = getMergedMesh(mmindex);
    guint4* nodeinfo = mm->getNodeInfo(); 

    unsigned int nso = mm->getNumVolumes() ;

    LOG(info) << msg 
              << " mmindex " << mmindex  
              << " volumes " << nso 
              ; 

    for(unsigned int i=0 ; i < nso ; i++)
    {
         guint4 ni = *(nodeinfo+i)  ;   
         const char* pv = getPVName(ni.z);
         const char* lv = getLVName(ni.z);
         printf( " %6d %6d %6d %6d lv %50s pv %s  \n", ni.x , ni.y, ni.z, ni.w, lv, pv );
    }
}




glm::mat4 GGeo::getTransform(int index) // TRY TO MOVE TO HUB
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





glm::vec4 GGeo::getCenterExtent(unsigned int target, unsigned int merged_mesh_index )
{
    assert(0); // moved to transform approach for torch targetting 

    GMergedMesh* mm = getMergedMesh(merged_mesh_index);
    assert(mm);

    glm::vec4 ce ; 
    if(merged_mesh_index == 0)
    {
        gfloat4 vce = mm->getCenterExtent(target); 
        ce.x = vce.x ; 
        ce.y = vce.y ; 
        ce.z = vce.z ; 
        ce.w = vce.w ; 
        print(ce, "GGeo::getCenterExtent target:%u", target);
    }
    else
    {
        float* transform = mm->getTransform(target);
        ce.x = *(transform + 4*3 + 0) ; 
        ce.y = *(transform + 4*3 + 1) ; 
        ce.z = *(transform + 4*3 + 2) ; 

        gfloat4 vce = mm->getCenterExtent(0); 
        ce.w = vce.w ;  
        // somewhat dodgy, should probably find the largest extent 
        // of all the local coordinate extents
    }
    return ce ; 
}




void GGeo::dumpTree(const char* msg)
{
    GMergedMesh* mm0 = getMergedMesh(0);

    // all these are full traverse counts, not reduced by selections or instancing
    unsigned int nso = mm0->getNumVolumes();  
    guint4* nodeinfo = mm0->getNodeInfo(); 

    unsigned int npv = m_nodelib->getNumPV();
    unsigned int nlv = m_nodelib->getNumLV(); 

    LOG(info) << msg 
              << " nso " << nso 
              << " npv " << npv 
              << " nlv " << nlv 
              << " nodeinfo " << (void*)nodeinfo
              ; 

    if( nso <= 10 || npv == 0 || nlv == 0 || nodeinfo == NULL )
    {
        LOG(warning) << "GGeo::dumpTree MISSING pvlist lvlist or nodeinfo OR few volume testing  " ; 
        return ;
    }
    else
    {
        assert(npv == nlv && nso == npv);
    }

    for(unsigned int i=0 ; i < nso ; i++)
    {
         guint4* info = nodeinfo + i ;  
         glm::ivec4 offnum = getNodeOffsetCount(i);

         //const char* pv = m_pvlist->getKey(i);
         //const char* lv = m_lvlist->getKey(i);

         const char* pv = m_nodelib->getPVName(i);
         const char* lv = m_nodelib->getLVName(i);


         printf(" %6u : nf %4d nv %4d id %6u pid %6d : %4d %4d %4d %4d  :%50s %50s \n", i, 
                    info->x, info->y, info->z, info->w,  offnum.x, offnum.y, offnum.z, offnum.w,
                    pv, lv ); 
    }
}






glm::ivec4 GGeo::getNodeOffsetCount(unsigned int index) // TODO: move into geolib
{
    GMergedMesh* mm0 = getMergedMesh(0);
    guint4* nodeinfo = mm0->getNodeInfo(); 
    unsigned int nso = mm0->getNumVolumes();   // poor name, means volumes
    assert(index < nso );

    glm::ivec4 offset ; 
    unsigned int cur_vert(0);
    unsigned int cur_face(0);

    for(unsigned int i=0 ; i < nso ; i++)
    {
        guint4* info = nodeinfo + i ;  
        if( i == index )
        {
           offset.x = cur_face ;   // cumulative sums of prior faces/verts in the buffer
           offset.y = cur_vert ;   //                  
           offset.z = info->x ;    // number faces/verts for this node
           offset.w = info->y ; 
           break ; 
        }
        cur_face += info->x ; 
        cur_vert += info->y ; 
    }
    return offset ; 
}


void GGeo::dumpVolume(unsigned int index, const char* msg)
{
    GMergedMesh* mm0 = getMergedMesh(0);
    unsigned int nvolume = mm0->getNumVolumes();  
    unsigned int nvert = mm0->getNumVertices();  
    unsigned int nface = mm0->getNumFaces();  
    LOG(info) << msg 
              << " nvolume " << nvolume
              << " nvert" << nvert
              << " nface " << nface
               ; 

    glm::ivec4 offnum = getNodeOffsetCount(index);
    LOG(info) << " nodeoffsetcount " 
              << " index " << index
              << " x " << offnum.x
              << " y " << offnum.y
              << " z " << offnum.z
              << " w " << offnum.w
              ;

    gfloat3* verts = mm0->getVertices();
    guint3* faces = mm0->getFaces(); 

    for(int i=0 ; i < offnum.z ; i++)
    {
        guint3* f = faces + offnum.x + i ;    // offnum.x is cumulative sum of prior volume face counts

        //  GMergedMesh::traverse  already does vertex index offsetting corresponding to the other volume meshes incorporated in the merge
        gfloat3* v0 = verts + f->x ; 
        gfloat3* v1 = verts + f->y ; 
        gfloat3* v2 = verts + f->z ; 

        glm::vec3 p0(v0->x, v0->y, v0->z);
        glm::vec3 p1(v1->x, v1->y, v1->z);
        glm::vec3 p2(v2->x, v2->y, v2->z);
        //glm::vec3 pc = (p0 + p1 + p2)/3.f ;
        glm::vec3 e0 = p1 - p0;
        glm::vec3 e1 = p0 - p2;
        glm::vec3 no = glm::normalize(glm::cross( e1, e0 ));

        printf(" i %3u f %4u %4u %4u : %10.3f %10.3f %10.3f    %10.3f %10.3f %10.3f     %10.3f %10.3f %10.3f   :  %10.3f %10.3f %10.3f \n", i, 
            f->x, f->y, f->z, 
            p0.x, p0.y, p0.z,
            p1.x, p1.y, p1.z,
            p2.x, p2.y, p2.z,
            no.x, no.y, no.z 
         ); 

    }
}


glm::vec4 GGeo::getFaceCenterExtent(unsigned int face_index, unsigned int volume_index, unsigned int mergedmesh_index )
{
   return getFaceRangeCenterExtent( face_index, face_index + 1 , volume_index, mergedmesh_index );
}

glm::vec4 GGeo::getFaceRangeCenterExtent(unsigned int face_index0, unsigned int face_index1, unsigned int volume_index, unsigned int mergedmesh_index )
{
    assert(mergedmesh_index == 0 && "instanced meshes not yet supported");
    GMergedMesh* mm = getMergedMesh(mergedmesh_index);
    assert(mm);
    unsigned int nvolume = mm->getNumVolumes();  
    assert(volume_index < nvolume);

    glm::ivec4 offnum = getNodeOffsetCount(volume_index);
    gfloat3* verts = mm->getVertices();
    guint3* faces = mm->getFaces(); 

    assert(int(face_index0) <  offnum.z );  
    assert(int(face_index1) <= offnum.z );   // face_index1 needs to go 1 beyond

    glm::vec3 lo( FLT_MAX,  FLT_MAX,  FLT_MAX);
    glm::vec3 hi(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    glm::vec3 centroid ; 

    unsigned int nface = face_index1 - face_index0 ; 
    for(unsigned int face_index=face_index0 ; face_index < face_index1 ; face_index++)
    {

        guint3* f = faces + offnum.x + face_index ; // offnum.x is cumulative sum of prior volume face counts within the merged mesh
        gfloat3* v = NULL ; 

        gfloat3* v0 = verts + f->x ;
        gfloat3* v1 = verts + f->y ;
        gfloat3* v2 = verts + f->z ;

        glm::vec3 p0(v0->x, v0->y, v0->z);
        glm::vec3 p1(v1->x, v1->y, v1->z);
        glm::vec3 p2(v2->x, v2->y, v2->z);

        centroid = centroid + p0 + p1 + p2  ; 

        for(unsigned int i=0 ; i < 3 ; i++)
        {
            switch(i)
            {
                case 0: v = v0 ; break ; 
                case 1: v = v1 ; break ; 
                case 2: v = v2 ; break ; 
            }

            lo.x = std::min( lo.x, v->x);
            lo.y = std::min( lo.y, v->y);
            lo.z = std::min( lo.z, v->z);

            hi.x = std::max( hi.x, v->x);
            hi.y = std::max( hi.y, v->y);
            hi.z = std::max( hi.z, v->z);
        }
    }

    glm::vec3 dim = hi - lo ; 

    float extent = 0.f ;
    extent = std::max( dim.x , extent ); 
    extent = std::max( dim.y , extent ); 
    extent = std::max( dim.z , extent ); 
    extent = extent / 2.0f  ;

    glm::vec4 ce ; 
    if( nface == 1 )
    {
       // for single face using avg matches OpenGL geom shader, and OptiX
        ce.x = centroid.x/3.f ; 
        ce.y = centroid.y/3.f ; 
        ce.z = centroid.z/3.f ; 
    }
    else
    {
       // for multiple faces use bbox center, as there are repeated vertices
       ce.x = (hi.x + lo.x)/2.f ; 
       ce.y = (hi.y + lo.y)/2.f ; 
       ce.z = (hi.z + lo.z)/2.f ; 
    }
    ce.w = extent ; 
 
    return ce ; 
}


bool GGeo::shouldMeshJoin(const GMesh* mesh)
{
    const char* shortname = mesh->getShortName();

    bool join = m_join_cfg && BStr::listHasKey(m_join_cfg, shortname, ",") ; 

    LOG(debug)<< "GGeo::shouldMeshJoin"
             << " shortname " << shortname
             << " join_cfg " << ( m_join_cfg ? m_join_cfg : "" )
             << " join " << join 
             ;

    return join ; 
}


GMesh* GGeo::invokeMeshJoin(GMesh* mesh)
{
    GMesh* result = mesh ; 
    bool join = shouldMeshJoin(mesh);
    if(join)
    {
        LOG(verbose) << "GGeo::invokeMeshJoin proceeding for " << mesh->getName() ; 

        result = (*m_join_imp)(mesh, m_ok); 

        result->setName(mesh->getName()); 
        result->setIndex(mesh->getIndex()); 
        result->updateBounds();
    }
    return result ; 
}







// pickface machinery must be here as GGeo cannot live in Opticks

void GGeo::setPickFace(std::string pickface)
{
    glm::ivec4 pf = givec4(pickface) ;
    setPickFace(pf);
}

glm::ivec4& GGeo::getPickFace()
{
    return m_composition->getPickFace();
}

void GGeo::setPickFace(const glm::ivec4& pickface) 
{
    m_composition->setPickFace(pickface);

    // gets called on recieving udp messages via boost bind done in CompositionCfg 
    LOG(info) << "GGeo::setPickFace " << gformat(pickface) ;    
    if(pickface.x > 0) 
    {    
        print(pickface, "GGeo::setPickFace face targetting");
        unsigned int face_index0= pickface.x ;
        unsigned int face_index1= pickface.y ;
        unsigned int volume_index= pickface.z ;
        unsigned int mesh_index = pickface.w ;

        //setFaceTarget(face_index, volume_index, mesh_index);
        setFaceRangeTarget(face_index0, face_index1, volume_index, mesh_index);
    }    
    else 
    {    
        LOG(warning) << "GGeo::setPickFace IGNORING " << gformat(pickface) ;    
    }    
}

void GGeo::setFaceTarget(unsigned int face_index, unsigned int volume_index, unsigned int mesh_index)
{
    glm::vec4 ce = getFaceCenterExtent(face_index, volume_index, mesh_index);
    bool autocam = false ; 
    m_composition->setCenterExtent(ce, autocam );
}


void GGeo::setFaceRangeTarget(unsigned int face_index0, unsigned int face_index1, unsigned int volume_index, unsigned int mesh_index)
{
    glm::vec4 ce = getFaceRangeCenterExtent(face_index0, face_index1, volume_index, mesh_index);
    bool autocam = false ;
    m_composition->setCenterExtent(ce, autocam );
}



void GGeo::set(const char* name, std::string& s)
{
    if(strcmp(name,PICKFACE)==0) setPickFace(s);
    else 
        printf("GGeo::set bad name %s\n", name);
}

std::string GGeo::get(const char* name)
{
   std::string s ;  
   if(strcmp(name,PICKFACE)==0) s = gformat(getPickFace()) ;
   else 
       printf("GGeo::get bad name %s\n", name);

   return s ;  
}

void GGeo::configure(const char* name, const char* value_)
{
    std::string value(value_);
    set(name, value);
}

const char* GGeo::PREFIX = "ggeo" ;
const char* GGeo::getPrefix()
{
   return PREFIX ;
}

std::vector<std::string> GGeo::getTags()
{
    std::vector<std::string> tags ;
    //tags.push_back(PICKFACE);
    return tags ;
}



void GGeo::anaEvent(OpticksEvent* evt)
{
    LOG(warning) << "GGeo::anaEvent" 
                 << " evt " << evt 
                 ;
}

