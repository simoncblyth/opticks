#include <cassert>
#include <cstdio>
#include <cstring>
#include <iomanip>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "BStr.hh"
#include "BMap.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NQuad.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "TorchStepNPY.hpp"
#include "NSensorList.hpp"
#include "NSensor.hpp"
#include "NLookup.hpp"
#include "NSlice.hpp"
#include "NScene.hpp"
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
#include "GSurLib.hh"
#include "GScintillatorLib.hh"
#include "GSourceLib.hh"


#include "GSolid.hh"
#include "GMesh.hh"
#include "GTreeCheck.hh"
#include "GTreePresent.hh"
#include "GColorizer.hh"
#include "GGeoTestConfig.hh"
#include "GGeoTest.hh"
#include "GPmt.hh"
#include "GScene.hh"

#include "GMergedMesh.hh"
#include "GItemIndex.hh"
#include "GItemList.hh"

#include "GGeo.hh"

#include "GGEO_BODY.hh"
#include "PLOG.hh"

#define BSIZ 50

const char* GGeo::CATHODE_MATERIAL = "Bialkali" ; 
const char* GGeo::PICKFACE = "pickface" ;



GGeo::GGeo(Opticks* opticks)
  :
   m_ok(opticks), 
   m_analytic(false),
   m_gltf(m_ok->getGLTF()),   
   m_composition(NULL), 
   m_treecheck(NULL), 
   m_loaded(false), 
   m_lookup(NULL), 
   m_meshlib(NULL),
   m_geolib(NULL),
   m_nodelib(NULL),
   m_bndlib(NULL),
   m_materiallib(NULL),
   m_surfacelib(NULL),
   m_surlib(NULL),
   m_scintillatorlib(NULL),
   m_sourcelib(NULL),
   m_pmt(NULL),
   m_colorizer(NULL),
   m_geotest(NULL),
   m_sensor_list(NULL),
   m_low(NULL),
   m_high(NULL),
   m_sensitive_count(0),
   m_volnames(false),
   m_cathode(NULL),
   m_join_cfg(NULL),
   m_loader_verbosity(0),
   m_mesh_verbosity(0),
   m_gscene(NULL)
{
   init(); 
}



// setLoaderImp : sets implementation that does the actual loading
// using a function pointer to the implementation 
// avoids ggeo-/GLoader depending on all the implementations

void GGeo::setLoaderImp(GLoaderImpFunctionPtr imp)
{
    m_loader_imp = imp ; 
}
void GGeo::setLoaderVerbosity(unsigned int verbosity)
{
    m_loader_verbosity = verbosity  ; 
}
unsigned int GGeo::getLoaderVerbosity()
{
    return m_loader_verbosity ;
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
unsigned int GGeo::getMeshVerbosity()
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

bool GGeo::isVolnames()
{
    return m_volnames ; 
}


void GGeo::addRaw(GMaterial* material)
{
    m_materials_raw.push_back(material);
}
void GGeo::addRaw(GBorderSurface* surface)
{
    m_border_surfaces_raw.push_back(surface);
}
void GGeo::addRaw(GSkinSurface* surface)
{
    m_skin_surfaces_raw.push_back(surface);
}


unsigned int GGeo::getNumMaterials()
{
    return m_materials.size();
}
unsigned int GGeo::getNumBorderSurfaces()
{
    return m_border_surfaces.size();
}
unsigned int GGeo::getNumSkinSurfaces()
{
    return m_skin_surfaces.size();
}
unsigned int GGeo::getNumRawMaterials()
{
    return m_materials_raw.size();
}
unsigned int GGeo::getNumRawBorderSurfaces()
{
    return m_border_surfaces_raw.size();
}
unsigned int GGeo::getNumRawSkinSurfaces()
{
    return m_skin_surfaces_raw.size();
}





GSkinSurface* GGeo::getSkinSurface(unsigned int index)
{
    return m_skin_surfaces[index];
}
GBorderSurface* GGeo::getBorderSurface(unsigned int index)
{
    return m_border_surfaces[index];
}



GBndLib* GGeo::getBndLib()
{
    return m_bndlib ; 
}

GMaterialLib* GGeo::getMaterialLib()
{
    return m_materiallib ; 
}
GSurfaceLib* GGeo::getSurfaceLib()
{
    return m_surfacelib ; 
}
GSurLib* GGeo::getSurLib()
{
    if(m_surlib == NULL) createSurLib();
    return m_surlib ; 
}

const char*  GGeo::getIdentifier()
{
    return "GGeo" ; 
}


GScintillatorLib* GGeo::getScintillatorLib()
{
    return m_scintillatorlib ; 
}
GSourceLib* GGeo::getSourceLib()
{
    return m_sourcelib ; 
}
GPmt* GGeo::getPmt()
{
    return m_pmt ; 
}

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
NSensorList* GGeo::getSensorList()
{
    return m_sensor_list ; 
}


gfloat3* GGeo::getLow()
{
   return m_low ; 
}
gfloat3* GGeo::getHigh()
{
   return m_high ; 
}


GTreeCheck* GGeo::getTreeCheck()
{
    return m_treecheck ;
}




GMaterial* GGeo::getCathode()
{
    return m_cathode ; 
}
void GGeo::setCathode(GMaterial* cathode)
{
    m_cathode = cathode ; 
}

void GGeo::addCathodeLV(const char* lv)
{
   m_cathode_lv.insert(lv);
}

unsigned int GGeo::getNumCathodeLV()
{
   return m_cathode_lv.size() ; 
}
const char* GGeo::getCathodeLV(unsigned int index)
{
    typedef std::unordered_set<std::string>::const_iterator UCI ; 
    UCI it = m_cathode_lv.begin() ; 
    std::advance( it, index );
    return it != m_cathode_lv.end() ? it->c_str() : NULL  ; 
}

void GGeo::dumpCathodeLV(const char* msg)
{
    printf("%s\n", msg);
    typedef std::unordered_set<std::string>::const_iterator UCI ; 
    for(UCI it=m_cathode_lv.begin() ; it != m_cathode_lv.end() ; it++)
    {
        printf("GGeo::dumpCathodeLV %s \n", it->c_str() ); 
    }
}


Opticks* GGeo::getOpticks()
{
    return m_ok ; 
}








void GGeo::init()
{
   LOG(trace) << "GGeo::init" ; 

   OpticksResource* resource = m_ok->getResource(); 
   const char* idpath = m_ok->getIdPath() ;

   LOG(trace) << "GGeo::init" 
             << " idpath " << ( idpath ? idpath : "NULL" )
             ; 

   assert(idpath && "GGeo::init idpath is required" );

   fs::path geocache(idpath); 

   bool cache_exists = fs::exists(geocache) && fs::is_directory(geocache) ;
   bool cache_requested = m_ok->isGeocache() ; 

   m_loaded = cache_exists && cache_requested ;

   LOG(trace) << "GGeo::init"
             << " idpath " << idpath
             << " cache_exists " << cache_exists 
             << " cache_requested " << cache_requested
             << " m_loaded " << m_loaded 
             ;

   const char* ctrl = resource->getCtrl() ;

   m_volnames = GGeo::ctrlHasKey(ctrl, "volnames");
 
   m_sensor_list = new NSensorList();

   m_sensor_list->load( idpath, "idmap");


   LOG(debug) << "GGeo::init loadSensorList " << m_sensor_list->description() ; 

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
   m_meshlib = new GMeshLib(m_ok, m_analytic);
   m_geolib = new GGeoLib(m_ok, m_analytic, m_bndlib );
   m_nodelib = new GNodeLib(m_ok, m_analytic ); 

   m_treecheck = new GTreeCheck(m_geolib, m_nodelib, m_ok->getSceneConfig() ) ;


/*
   if(resource->isJuno())
   {
       m_treecheck->setVertexMin(10);  
       //m_treecheck->setVertexMin(250);
   } 
*/
   //GColorizer::Style_t style  = GColorizer::SURFACE_INDEX ;  // rather grey 
   GColorizer::Style_t style = GColorizer::PSYCHEDELIC_NODE ;
   OpticksColors* colors = getColors();

   m_colorizer = new GColorizer( m_nodelib, m_geolib, m_bndlib, colors, style ); // colorizer needs full tree, so pre-cache only 


   m_scintillatorlib  = new GScintillatorLib(m_ok);
   m_sourcelib  = new GSourceLib(m_ok);



   LOG(trace) << "GGeo::init DONE" ; 
}



void GGeo::add(GMaterial* material)
{
    m_materiallib->add(material);
    m_materials.push_back(material);
    addToIndex((GPropertyMap<float>*)material);
}
void GGeo::add(GBorderSurface* surface)
{
    m_surfacelib->add(surface);
    m_border_surfaces.push_back(surface);
    addToIndex((GPropertyMap<float>*)surface);
}
void GGeo::add(GSkinSurface* surface)
{
    LOG(trace) << "GGeo::add(GSkinSurface*) " << ( surface ? surface->getName() : "NULL" ) ;

    m_surfacelib->add(surface);
    m_skin_surfaces.push_back(surface);

    addToIndex((GPropertyMap<float>*)surface);
}


GGeoLib* GGeo::getGeoLib()
{
    return m_geolib ; 
}
GNodeLib* GGeo::getNodeLib()
{
    return m_nodelib ; 
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

GMergedMesh* GGeo::getMergedMesh(unsigned int index)
{
    GGeoLib* geolib = getGeoLib() ;
    assert(geolib);

    GMergedMesh* mm = geolib->getMergedMesh(index);

    unsigned int meshverbosity = getMeshVerbosity() ; 

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

    loadAnalyticPmt();

    if( gltf >= 10 )
    {
        LOG(info) << "GGeo::loadGeometry DEBUFFING loadAnalyticFromGLTF " ; 
        loadAnalyticFromGLTF();
    }

    setupLookup();
    setupColors();
    setupTyp();
    LOG(info) << "GGeo::loadGeometry DONE" ; 
}

void GGeo::loadFromG4DAE()
{
    LOG(trace) << "GGeo::loadFromG4DAE START" ; 

    int rc = (*m_loader_imp)(this);   //  imp set in OpticksGeometry::loadGeometryBase, m_ggeo->setLoaderImp(&AssimpGGeo::load); 

    assert(rc == 0);

    prepareScintillatorLib();

    prepareMeshes();

    prepareVertexColors();

    LOG(trace) << "GGeo::loadFromG4DAE DONE" ; 
}


void GGeo::loadAnalyticFromGLTF()
{
    LOG(info) << "GGeo::loadAnalyticFromGLTF START" ; 
    if(!m_ok->isGLTF()) return ; 
#ifdef WITH_YoctoGL

    bool loaded = false ; 
    m_gscene = new GScene(m_ok, this, loaded); // GGeo needed for m_bndlib 

#else
    LOG(fatal) << "GGeo::loadAnalyticFromGLTF requires YoctoGL external " ; 
    assert(0);
#endif
    LOG(info) << "GGeo::loadAnalyticFromGLTF DONE" ; 
}


void GGeo::save()
{
    LOG(info) << "GGeo::save" ;
    m_geolib->dump("GGeo::save.geolib");

    m_geolib->save();
    m_meshlib->save();
    m_nodelib->save();
    m_materiallib->save();
    m_surfacelib->save();
    m_scintillatorlib->save();
    m_sourcelib->save();
    m_bndlib->save();  
}

void GGeo::saveAnalytic()
{ 
    LOG(info) << "GGeo::saveAnalytic" ;
    m_gscene->save();
}
 
void GGeo::loadFromCache()
{   
    LOG(info) << "GGeo::loadFromCache START" ; 
    bool constituents = true ; 
    m_bndlib = GBndLib::load(m_ok, constituents);    // interpolation potentially happens in here

    // GBndLib is persisted via index buffer, not float buffer
    m_materiallib = m_bndlib->getMaterialLib();
    m_surfacelib = m_bndlib->getSurfaceLib();

    m_scintillatorlib  = GScintillatorLib::load(m_ok);
    m_sourcelib  = GSourceLib::load(m_ok);

    bool analytic = false ; 
    m_geolib = GGeoLib::Load(m_ok, analytic, m_bndlib);
    m_nodelib = GNodeLib::Load(m_ok, analytic);        
    m_meshlib = GMeshLib::Load(m_ok, analytic);


    LOG(info) << "GGeo::loadFromCache DONE" ; 
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



/*
GSolid* GGeo::getSolidAnalytic(unsigned idx)
{
    return m_nodelib_analytic ? m_nodelib_analytic->getSolid(idx) : NULL ; 
}
*/



void GGeo::createSurLib()
{
/*
    This is deferred until called upon by CG4/CGeometry so any test geometry mesh0 modifications 
    will have been done already when called...

    frame #4: 0x0000000101d1cfce libGGeo.dylib`GGeo::createSurLib(this=0x0000000109900410) + 46 at GGeo.cc:637
    frame #5: 0x0000000101d1cf8e libGGeo.dylib`GGeo::getSurLib(this=0x0000000109900410) + 46 at GGeo.cc:259
    frame #6: 0x0000000103e2eebb libcfg4.dylib`CGeometry::CGeometry(this=0x000000010e38fa60, hub=0x000000010980b170) + 91 at CGeometry.cc:33
    frame #7: 0x0000000103e2f56d libcfg4.dylib`CGeometry::CGeometry(this=0x000000010e38fa60, hub=0x000000010980b170) + 29 at CGeometry.cc:43
    frame #8: 0x0000000103ec8cc9 libcfg4.dylib`CG4::CG4(this=0x000000010e153de0, hub=0x000000010980b170) + 217 at CG4.cc:113
    frame #9: 0x0000000103ec919d libcfg4.dylib`CG4::CG4(this=0x000000010e153de0, hub=0x000000010980b170) + 29 at CG4.cc:134
    frame #10: 0x0000000103faffb3 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe660, argc=21, argv=0x00007fff5fbfe748) + 547 at OKG4Mgr.cc:35
    frame #11: 0x0000000103fb0203 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe660, argc=21, argv=0x00007fff5fbfe748) + 35 at OKG4Mgr.cc:41
    frame #12: 0x00000001000139be OKG4Test`main(argc=21, argv=0x00007fff5fbfe748) + 1486 at OKG4Test.cc:56
*/

    if(m_surlib)
    {
        LOG(warning) << "recreating GSurLib" ; 
        delete m_surlib ; 
    }
    else
    {
        LOG(info) << "deferred creation of GSurLib " ; 
    }

    m_surlib = new GSurLib(this) ; 
    //m_surlib->dump("GGeo::createSurLib");
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
    LOG(trace) << "GGeo::setupColors" ; 

    //OpticksFlags* flags = m_ok->getFlags();

    std::vector<unsigned int>& material_codes = m_materiallib->getAttrNames()->getColorCodes() ; 
    std::vector<unsigned int>& flag_codes     = m_ok->getFlagNames()->getColorCodes() ; 

    OpticksColors* colors = m_ok->getColors();

    colors->setupCompositeColorBuffer( material_codes, flag_codes  );

    LOG(trace) << "GGeo::setupColors DONE" ; 
}


void GGeo::loadAnalyticPmt()
{
    NSlice* slice = m_ok->getAnalyticPMTSlice();

    unsigned apmtidx = m_ok->getAnalyticPMTIndex();

    m_pmt = GPmt::load( m_ok, m_bndlib, apmtidx, slice ); 

    LOG(info) << "GGeo::loadAnalyticPmt"
              << " AnalyticPMTIndex " << apmtidx
              << " AnalyticPMTSlice " << ( slice ? slice->description() : "ALL" )
              << " m_pmt " << m_pmt 
              << " Path " << ( m_pmt ? m_pmt->getPath() : "-" ) 
              ;  

    if(m_pmt)
    {
        LOG(trace) << "GGeo::loadAnalyticPmt SUCCEEDED "
                   << m_pmt->getPath()   
                    ;
    }
}




void GGeo::modifyGeometry(const char* config)
{
    // NB only invoked with test option : "op --test" 
    //   controlled from OpticksGeometry::loadGeometry 

    GGeoTestConfig* gtc = new GGeoTestConfig(config);

    assert(m_geotest == NULL);
    m_geotest = new GGeoTest(m_ok, gtc, this);
    m_geotest->modifyGeometry();
}






bool GGeo::ctrlHasKey(const char* ctrl, const char* key)
{
    return BStr::listHasKey(ctrl, key, ",");
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
              << " so " << m_nodelib->getNumSolids()
              << " mt " << m_materials.size()
              << " bs " << m_border_surfaces.size()
              << " ss " << m_skin_surfaces.size()
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

    for(unsigned int ibs=0 ; ibs < m_border_surfaces.size()  ; ibs++ )
    {
        GBorderSurface* bs = m_border_surfaces[ibs];
        snprintf(mbuf,BSIZ, "%s bs %u", msg, ibs);
        bs->Summary(mbuf);
    }
    for(unsigned int iss=0 ; iss < m_skin_surfaces.size()  ; iss++ )
    {
        GSkinSurface* ss = m_skin_surfaces[iss];
        snprintf(mbuf,BSIZ, "%s ss %u", msg, iss);
        ss->Summary(mbuf);
    }
    for(unsigned int imat=0 ; imat < m_materials.size()  ; imat++ )
    {
        GMaterial* mat = m_materials[imat];
        snprintf(mbuf,BSIZ, "%s mt %u", msg, imat);
        mat->Summary(mbuf);
    }

    /*
    for(unsigned int isol=0 ; isol < m_solids.size()  ; isol++ )
    {
        GSolid* sol = m_solids[isol];
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
unsigned GGeo::getNumMeshes()
{
    return m_meshlib->getNumMeshes(); 
}
GItemIndex* GGeo::getMeshIndex()
{
    return m_meshlib->getMeshIndex() ; 
}
GMesh* GGeo::getMesh(unsigned int aindex)
{
    return m_meshlib->getMesh(aindex);
}  
void GGeo::add(GMesh* mesh)
{
    m_meshlib->add(mesh);
}





// via GNodeLib

unsigned GGeo::getNumSolids()
{
    return m_nodelib->getNumSolids();
}
void GGeo::add(GSolid* solid)
{
    m_nodelib->add(solid);
}
GSolid* GGeo::getSolid(unsigned index)
{
    return m_nodelib->getSolid(index);
}
GSolid* GGeo::getSolidSimple(unsigned int index)
{
    return m_nodelib->getSolidSimple(index);
}
const char* GGeo::getPVName(unsigned int index)
{
    return m_nodelib->getPVName(index);
}
const char* GGeo::getLVName(unsigned int index)
{
    return m_nodelib->getLVName(index);
}
GNode* GGeo::getNode(unsigned index)
{
    return m_nodelib->getNode(index);
}





void GGeo::dumpRaw(const char* msg)
{
    printf("%s\n", msg);     
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++)
    {
        GMaterial* mat = m_materials_raw[i];
        mat->Summary();
    }
}





#if 0

// cannot do this check any more in GBoundary approach 

void GGeo::materialConsistencyCheck()
{
    GSolid* solid = getSolid(0);
    assert(solid);
    unsigned int nok = materialConsistencyCheck(solid);
    printf("GGeo::materialConsistencyCheck nok %u \n", nok );
}

unsigned int GGeo::materialConsistencyCheck(GSolid* solid)
{
    assert(solid);
    //solid->Summary(NULL);

    GSolid* parent = dynamic_cast<GSolid*>(solid->getParent()) ; 

    unsigned int nok = 0 ;
    if(parent)
    {
        assert(parent->getInnerMaterial() == solid->getOuterMaterial());
        nok += 1 ;
    } 
    else
    {
        assert(solid->getIndex() == 0); 
    } 

    for(unsigned int i=0 ; i < solid->getNumChildren() ; i++)
    {
        GSolid* child = dynamic_cast<GSolid*>(solid->getChild(i)) ;
        assert(child); 
        nok += materialConsistencyCheck(child);
    }
    return nok ;
}

#endif





GMaterial* GGeo::getMaterial(unsigned int aindex)
{
    GMaterial* mat = NULL ; 
    for(unsigned int i=0 ; i < m_materials.size() ; i++ )
    { 
        if(m_materials[i]->getIndex() == aindex )
        {
            mat = m_materials[i] ; 
            break ; 
        }
    }
    return mat ;
}


GPropertyMap<float>* GGeo::findRawMaterial(const char* shortname)
{
    GMaterial* mat = NULL ; 
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++ )
    { 
        std::string sn = m_materials_raw[i]->getShortNameString();
        //printf("GGeo::findRawMaterial %d %s \n", i, sn.c_str()); 
        if(strcmp(sn.c_str(), shortname)==0)
        {
            mat = m_materials_raw[i] ; 
            break ; 
        }
    }
    return (GPropertyMap<float>*)mat ;
}


void GGeo::addToIndex(GPropertyMap<float>* psrc)
{
    unsigned int pindex = psrc->getIndex();
    if(pindex < UINT_MAX)
    {
         if(m_index.count(pindex) == 0) 
               m_index[pindex] = psrc->getShortName(); 
         else
               assert(strcmp(m_index[pindex].c_str(), psrc->getShortName()) == 0);
    }
}


void  GGeo::dumpIndex(const char* msg)
{
    printf("%s\n", msg);
    for(Index_t::iterator it=m_index.begin() ; it != m_index.end() ; it++)
         printf("  %3u :  %s \n", it->first, it->second.c_str() );
}



GProperty<float>* GGeo::findRawMaterialProperty(const char* shortname, const char* propname)
{
    GPropertyMap<float>* mat = findRawMaterial(shortname);

    GProperty<float>* prop = mat->getProperty(propname);
    prop->Summary();

    // hmm should have permanent slot in idpath 
    return prop ;   
}








GSkinSurface* GGeo::findSkinSurface(const char* lv)
{
    GSkinSurface* ss = NULL ; 
    for(unsigned int i=0 ; i < m_skin_surfaces.size() ; i++ )
    {
         GSkinSurface* s = m_skin_surfaces[i];
         if(s->matches(lv))   
         {
            ss = s ; 
            break ; 
         } 
    }
    return ss ;
}

GBorderSurface* GGeo::findBorderSurface(const char* pv1, const char* pv2)
{
    GBorderSurface* bs = NULL ; 
    for(unsigned int i=0 ; i < m_border_surfaces.size() ; i++ )
    {
         GBorderSurface* s = m_border_surfaces[i];
         if(s->matches(pv1,pv2))   
         {
            bs = s ; 
            break ; 
         } 
    }
    return bs ;
}



void GGeo::dumpRawSkinSurface(const char* name)
{
    LOG(info) << "GGeo::dumpRawSkinSurface " << name ; 

    GSkinSurface* ss = NULL ; 
    unsigned int n = getNumRawSkinSurfaces();
    for(unsigned int i = 0 ; i < n ; i++)
    {
        ss = m_skin_surfaces_raw[i];
        ss->Summary("GGeo::dumpRawSkinSurface", 10); 
    }
}

void GGeo::dumpRawBorderSurface(const char* name)
{
    LOG(info) << "GGeo::dumpRawBorderSurface " << name ; 
    GBorderSurface* bs = NULL ; 
    unsigned int n = getNumRawBorderSurfaces();
    for(unsigned int i = 0 ; i < n ; i++)
    {
        bs = m_border_surfaces_raw[i];
        bs->Summary("GGeo::dumpRawBorderSurface", 10); 
    }
}




void GGeo::traverse(const char* msg)
{
    LOG(info) << msg ; 
    traverse( getSolid(0), 0 );
}


void GGeo::traverse( GNode* node, unsigned int depth)
{
    GSolid* solid = dynamic_cast<GSolid*>(node) ;

    NSensor* sensor = solid->getSensor(); 

    if(sensor)
         LOG(debug) << "GGeo::traverse " 
                   << " nodeIndex " << node->getIndex()
                   << sensor->description() 
                   ; 

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1);
}



void GGeo::dumpRawMaterialProperties(const char* msg)
{
    printf("%s\n", msg);     
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++)
    {
        GMaterial* mat = m_materials_raw[i];
        //mat->Summary();
        std::cout << std::setw(30) << mat->getShortName()
                  << " keys: " << mat->getKeysString()
                  << std::endl ; 
    }
}



void GGeo::prepareMaterialLib()
{
    LOG(trace) << "GGeo::prepareMaterialLib " ; 

    GMaterialLib* mlib = getMaterialLib() ;
   
    mlib->addTestMaterials(); 
}

void GGeo::prepareSurfaceLib()
{
    LOG(trace) << "GGeo::prepareSurfaceLib " ; 

    GSurfaceLib* slib = getSurfaceLib() ;
   
    slib->addPerfectSurfaces(); 
}



void GGeo::prepareScintillatorLib()
{
    LOG(trace) << "GGeo::prepareScintillatorLib " ; 

    findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB"); 

    unsigned int nscint = getNumScintillatorMaterials() ;

    if(nscint == 0)
    {
        LOG(warning) << "GGeo::prepareScintillatorLib found no scintillator materials  " ; 
    }
    else
    {
        LOG(info) << "GGeo::prepareScintillatorLib found " << nscint << " scintillator materials  " ; 

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
    m_scintillators_raw = getRawMaterialsWithProperties(props, ",");
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



void GGeo::prepareMeshes()
{
    bool instanced = m_ok->isInstanced();

    LOG(trace) << "GGeo::prepareMeshes START" 
              << " instanced " << instanced 
              ;
    unsigned verbosity = 0 ; 

    if(instanced)
    { 
        bool deltacheck = true ; 
        m_treecheck->createInstancedMergedMeshes(deltacheck, verbosity);   // GTreeCheck::createInstancedMergedMeshes

    }
    else
    {
        LOG(warning) << "GGeo::prepareMeshes instancing inhibited " ;
        GNode* root = getNode(0);
        m_geolib->makeMergedMesh(0, NULL, root, verbosity);  // ridx:0 rbase:NULL 
        // ^^^^  precache never needs analytic geolib ?
    }
    LOG(trace) << "GGeo::prepareMeshes DONE" ;
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
    LOG(trace) << "GGeo::prepareVertexColors START" ;
    m_colorizer->writeVertexColors();
    LOG(trace) << "GGeo::prepareVertexColors DONE " ;
}






void GGeo::findCathodeMaterials(const char* props)
{
    m_cathodes_raw = getRawMaterialsWithProperties(props, ",");
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












std::vector<GMaterial*> GGeo::getRawMaterialsWithProperties(const char* props, const char* delim)
{
    std::vector<std::string> elem ;
    boost::split(elem, props, boost::is_any_of(delim));

    std::vector<GMaterial*>  selected ; 
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++)
    {
        GMaterial* mat = m_materials_raw[i];
        unsigned int found(0);
        for(unsigned int p=0 ; p < elem.size() ; p++)
        { 
           if(mat->hasProperty(elem[p].c_str())) found+=1 ;        
        }
        if(found == elem.size()) selected.push_back(mat);
    }
    return selected ;  
}





void GGeo::countMeshUsage(unsigned int meshIndex, unsigned int nodeIndex, const char* /*lv*/, const char* /*pv*/)
{

     // called during GGeo creation from: void AssimpGGeo::convertStructure(GGeo* gg)
     //printf("GGeo::countMeshUsage %d %d %s %s \n", meshIndex, nodeIndex, lv, pv);
     m_mesh_usage[meshIndex] += 1 ; 
     m_mesh_nodes[meshIndex].push_back(nodeIndex); 
}


std::map<unsigned int, unsigned int>& GGeo::getMeshUsage()
{
    return m_mesh_usage ; 
}
std::map<unsigned int, std::vector<unsigned int> >& GGeo::getMeshNodes()
{
    return m_mesh_nodes ; 
}


void GGeo::reportMeshUsage(const char* msg)
{
     printf("%s\n", msg);
     unsigned int tv(0) ; 
     typedef std::map<unsigned int, unsigned int>::const_iterator MUUI ; 
     for(MUUI it=m_mesh_usage.begin() ; it != m_mesh_usage.end() ; it++)
     {
         unsigned int meshIndex = it->first ; 
         unsigned int nodeCount = it->second ; 
 
         GMesh* mesh = getMesh(meshIndex);
         const char* meshName = mesh->getName() ; 
         unsigned int nv = mesh->getNumVertices() ; 
         unsigned int nf = mesh->getNumFaces() ; 

         printf("  %4d (v%5d f%5d) : %6d : %7d : %s \n", meshIndex, nv, nf, nodeCount, nodeCount*nv, meshName);
         tv += nodeCount*nv ; 
     }
     printf(" tv : %7d \n", tv);
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

    unsigned int nso = mm->getNumSolids() ;

    LOG(info) << msg 
              << " mmindex " << mmindex  
              << " solids " << nso 
              ; 

    for(unsigned int i=0 ; i < nso ; i++)
    {
         guint4 ni = *(nodeinfo+i)  ;   
         const char* pv = getPVName(ni.z);
         const char* lv = getLVName(ni.z);
         printf( " %6d %6d %6d %6d lv %50s pv %s  \n", ni.x , ni.y, ni.z, ni.w, lv, pv );
    }

}




glm::mat4 GGeo::getTransform(int index)
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
    unsigned int nso = mm0->getNumSolids();  
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
        LOG(warning) << "GGeo::dumpTree MISSING pvlist lvlist or nodeinfo OR few solid testing  " ; 
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
    unsigned int nso = mm0->getNumSolids();   // poor name, means volumes
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
    unsigned int nsolid = mm0->getNumSolids();  
    unsigned int nvert = mm0->getNumVertices();  
    unsigned int nface = mm0->getNumFaces();  
    LOG(info) << msg 
              << " nsolid " << nsolid
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
        guint3* f = faces + offnum.x + i ;    // offnum.x is cumulative sum of prior solid face counts

        //  GMergedMesh::traverse  already does vertex index offsetting corresponding to the other solid meshes incorporated in the merge
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


glm::vec4 GGeo::getFaceCenterExtent(unsigned int face_index, unsigned int solid_index, unsigned int mergedmesh_index )
{
   return getFaceRangeCenterExtent( face_index, face_index + 1 , solid_index, mergedmesh_index );
}

glm::vec4 GGeo::getFaceRangeCenterExtent(unsigned int face_index0, unsigned int face_index1, unsigned int solid_index, unsigned int mergedmesh_index )
{
    assert(mergedmesh_index == 0 && "instanced meshes not yet supported");
    GMergedMesh* mm = getMergedMesh(mergedmesh_index);
    assert(mm);
    unsigned int nsolid = mm->getNumSolids();  
    assert(solid_index < nsolid);

    glm::ivec4 offnum = getNodeOffsetCount(solid_index);
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

        guint3* f = faces + offnum.x + face_index ; // offnum.x is cumulative sum of prior solid face counts within the merged mesh
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


bool GGeo::shouldMeshJoin(GMesh* mesh)
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
        LOG(trace) << "GGeo::invokeMeshJoin proceeding for " << mesh->getName() ; 

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
        unsigned int solid_index= pickface.z ;
        unsigned int mesh_index = pickface.w ;

        //setFaceTarget(face_index, solid_index, mesh_index);
        setFaceRangeTarget(face_index0, face_index1, solid_index, mesh_index);
    }    
    else 
    {    
        LOG(warning) << "GGeo::setPickFace IGNORING " << gformat(pickface) ;    
    }    
}

void GGeo::setFaceTarget(unsigned int face_index, unsigned int solid_index, unsigned int mesh_index)
{
    glm::vec4 ce = getFaceCenterExtent(face_index, solid_index, mesh_index);
    bool autocam = false ; 
    m_composition->setCenterExtent(ce, autocam );
}


void GGeo::setFaceRangeTarget(unsigned int face_index0, unsigned int face_index1, unsigned int solid_index, unsigned int mesh_index)
{
    glm::vec4 ce = getFaceRangeCenterExtent(face_index0, face_index1, solid_index, mesh_index);
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



