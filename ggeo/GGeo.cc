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

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <csignal>

#include "SSys.hh"
#include "SLog.hh"
#include "BStr.hh"
#include "BMap.hh"
#include "BTxt.hh"

// npy-
#include "NGLM.hpp"
#include "NGLMExt.hpp"
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
#include "OpticksIdentity.hh"
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

#include "GParts.hh"


#include "GGeoGLTF.hh"
#include "GVolume.hh"
#include "GMesh.hh"
#include "GInstancer.hh"
#include "GTreePresent.hh"
#include "GColorizer.hh"
#include "GPmtLib.hh"

#include "GMergedMesh.hh"
#include "GItemIndex.hh"
#include "GItemList.hh"

#include "GGeo.hh"

#include "GGEO_BODY.hh"
#include "PLOG.hh"

#define BSIZ 50


const plog::Severity GGeo::LEVEL = PLOG::EnvLevel("GGeo", "DEBUG")  ; 

const char* GGeo::PICKFACE = "pickface" ;

GGeo* GGeo::fInstance = NULL ; 

GGeo* GGeo::GetInstance() // static
{
    return fInstance ;    
}

GGeo* GGeo::Load(Opticks* ok) // static
{
    if(!ok->isConfigured()) ok->configure(); 
    bool live = false ; 
    GGeo* ggeo = new GGeo(ok, live);
    ggeo->loadFromCache();
    ggeo->dumpStats("GGeo::Load");
    return ggeo ;  
}



/**
GGeo::GGeo
-------------

live=true instanciation only used from G4Opticks::translateGeometry 

**/


GGeo::GGeo(Opticks* ok, bool live)
  :
   m_log(new SLog("GGeo::GGeo","",verbose)),
   m_ok(ok), 
   m_enabled_legacy_g4dae(ok->isEnabledLegacyG4DAE()),   // --enabled_legacy_g4dae
   m_live(live),    // live=false  by default, only true when translated from Geant4 tree 
   m_composition(NULL), 
   m_instancer(NULL), 
   m_loaded_from_cache(false), 
   m_prepared(false), 
   m_gdmlauxmeta(NULL),
   m_loadedcachemeta(NULL),
   m_lv2sd(NULL),
   m_lv2mt(NULL),
   m_origin_gdmlpath(NULL),
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
#ifdef OLD_BOUNDS
   m_low(NULL),
   m_high(NULL),
#endif
   m_sensitive_count(0),
   m_join_cfg(NULL),
   m_mesh_verbosity(0),
#ifdef OLD_SCENE
   m_gscene(NULL),
#endif
   m_placeholder_last(0)
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

void GGeo::setGDMLAuxMeta(NMeta* gdmlauxmeta)
{
    m_gdmlauxmeta = gdmlauxmeta ; 
}
NMeta* GGeo::getGDMLAuxMeta() const 
{
    return m_gdmlauxmeta ; 
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
GMeshLib*         GGeo::getMeshLib() const { return m_meshlib ; }
 
GBndLib*          GGeo::getBndLib() const { return m_bndlib ; } 
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


GInstancer* GGeo::getInstancer() const 
{
    return m_instancer ;
}






Opticks* GGeo::getOpticks() const 
{
    return m_ok ; 
}



/**
GGeo::init
-------------

When the geocache exists and are configured to use it 
this will not instanciate the libs as those will subsequently 
be loaded from geocache.

G4Opticks::translateGeometry inhibits loading cached libs 
via setting m_live in ctor

**/

void GGeo::init()
{
    LOG(LEVEL) << "[" ; 
    const char* idpath = m_ok->getIdPath() ;
    LOG(LEVEL) << " idpath " << ( idpath ? idpath : "NULL" ) ; 
    assert(idpath && "GGeo::init idpath is required" );


    bool geocache_available = m_ok->isGeocacheAvailable() ;   // cache exists and is enabled  

    bool will_load_libs = geocache_available && (m_live == false ); 
    bool will_init_libs = !will_load_libs ; 

    m_loaded_from_cache = will_load_libs ; 

    LOG(LEVEL) 
        << " idpath " << idpath
        << " geocache_available  " << geocache_available
        << " m_live " << m_live 
        << " will_init_libs " << will_init_libs 
        ;

    if(will_init_libs)
    {
        initLibs() ;  
    }

    LOG(LEVEL) << "]" ; 
}


/**
GGeo::initLibs
----------------

This only happens when operating pre-cache

**/

void GGeo::initLibs()
{
   LOG(LEVEL) << "[ pre-cache initializing libs " ; 

   m_bndlib = new GBndLib(m_ok);
   m_materiallib = new GMaterialLib(m_ok);
   m_surfacelib  = new GSurfaceLib(m_ok);

   m_bndlib->setMaterialLib(m_materiallib);
   m_bndlib->setSurfaceLib(m_surfacelib);

   m_meshlib = new GMeshLib(m_ok);
   m_geolib = new GGeoLib(m_ok, m_bndlib );
   m_nodelib = new GNodeLib(m_ok); 

   m_instancer = new GInstancer(m_ok, this ) ;


   GColorizer::Style_t style = GColorizer::PSYCHEDELIC_NODE ;
   OpticksColors* colors = getColors();

   m_colorizer = new GColorizer( m_nodelib, m_geolib, m_bndlib, colors, style ); // colorizer needs full tree, so pre-cache only 

   m_scintillatorlib  = new GScintillatorLib(m_ok);
   m_sourcelib  = new GSourceLib(m_ok);

   m_pmtlib = NULL ; 

   LOG(LEVEL) << "]" ; 
}




// via Opticks

const char*      GGeo::getIdPath() { return m_ok->getIdPath(); } 
OpticksColors*   GGeo::getColors() { return m_ok->getColors() ; } 
OpticksFlags*    GGeo::getFlags() { return m_ok->getFlags(); }
OpticksAttrSeq*  GGeo::getFlagNames() { return m_ok->getFlagNames(); } 
OpticksResource* GGeo::getResource() { return m_ok->getResource(); } 


// via GMaterialLib

void     GGeo::add(GMaterial* material) { m_materiallib->add(material); } 
void     GGeo::addRaw(GMaterial* material) { m_materiallib->addRaw(material); } 
unsigned GGeo::getNumMaterials() const { return m_materiallib->getNumMaterials(); } 
unsigned GGeo::getNumRawMaterials() const { return m_materiallib->getNumRawMaterials(); } 

// via GBndLib

unsigned int GGeo::getMaterialLine(const char* shortname) { return m_bndlib->getMaterialLine(shortname); }
std::string  GGeo::getSensorBoundaryReport() const { return m_bndlib->getSensorBoundaryReport() ; }



// via GGeoLib

unsigned int GGeo::getNumMergedMesh() const 
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
    if(mm) mm->setVerbosity(meshverbosity);
    return mm ; 
}



bool GGeo::isLoadedFromCache() const { return m_loaded_from_cache ; } 

/**
GGeo::loadGeometry
--------------------


**/


void GGeo::loadGeometry()
{
    LOG(LEVEL) << "[" << " m_loaded_from_cache " << m_loaded_from_cache ; 

    if(!m_loaded_from_cache)
    {
        LOG(fatal) << "MISSING geocache : create one from GDML with geocache-;geocache-create " ; 
        assert(0);  
    }
    else
    {
        loadFromCache();
    } 

    // HMM : this not done in direct route ?
    setupLookup();
    setupColors();
    setupTyp();

    LOG(LEVEL) << "]" ; 
}


/**
GGeo::loadFromCache
---------------------
**/

void GGeo::loadFromCache()
{   
    LOG(LEVEL) << "[ " << m_ok->getIdPath()  ; 

    bool constituents = true ; 
    m_bndlib = GBndLib::load(m_ok, constituents);    // interpolation potentially happens in here

    // GBndLib is persisted via index buffer, not float buffer
    m_materiallib = m_bndlib->getMaterialLib();
    m_surfacelib = m_bndlib->getSurfaceLib();

    m_scintillatorlib  = GScintillatorLib::load(m_ok);
    m_sourcelib  = GSourceLib::load(m_ok);

    m_geolib = GGeoLib::Load(m_ok, m_bndlib);
    m_nodelib = GNodeLib::Load(m_ok );        
    m_meshlib = GMeshLib::Load(m_ok );

    postLoadFromCache(); 

    LOG(LEVEL) << "]" ; 
}


/**
GGeo::postLoadFromCache
-------------------------

Invoked from GGeo::loadFromCache immediately after loading the libs 
**/

void GGeo::postLoadFromCache()
{
    loadCacheMeta();

    close();                  // formerly OpticksHub::loadGeometry
    deferredCreateGParts();   // formerly OpticksHub::init   <-- this is needed for live running also  
}

/**
GGeo::postDirectTranslation
-------------------------------

Invoked from G4Opticks::translateGeometry after the X4PhysicalVolume conversion
for live running or from okg4/tests/OKX4Test.cc main for geocache-create.

**/


void GGeo::postDirectTranslation()
{
    LOG(LEVEL) << "[" ; 

    prepare();     // instances are formed here     

    LOG(LEVEL) << "( GBndLib::fillMaterialLineMap " ; 
    GBndLib* blib = getBndLib();
    blib->fillMaterialLineMap();
    LOG(LEVEL) << ") GBndLib::fillMaterialLineMap " ; 

    LOG(LEVEL) << "( GGeo::save " ; 
    save();
    LOG(LEVEL) << ") GGeo::save " ; 


    deferredCreateGParts();  

    postDirectTranslationDump(); 

    LOG(LEVEL) << "]" ; 
}



void GGeo::postDirectTranslationDump() const 
{
    LOG(LEVEL) << "[" ; 
    reportMeshUsage();

    if(m_ok->isDumpSensor())
    {
        dumpSensorVolumes("GGeo::postDirectTranslationDump --dumpsensor ");  
    }
    else
    {
        LOG(info) << reportSensorVolumes("GGeo::postDirectTranslationDump NOT --dumpsensor");  
    }

    LOG(LEVEL) << "]" ; 
}




bool GGeo::isPrepared() const { return m_prepared ; }


/**
GGeo::prepare
---------------
    
Prepare is needed prior to saving to geocache or GPU upload by OGeo, it 
is invoked for example from GGeo::postDirectTranslation. 

**/

void GGeo::prepare()
{
    //std::raise(SIGINT); 

    LOG(info) << "[" ; 
    assert( m_instancer && "GGeo::prepare can only be done pre-cache when the full node tree is available"); 

    assert( m_prepared == false && "have prepared already" ); 
    m_prepared = true ; 

    //TODO: implement prepareSensorSurfaces() and invoke from here 

    LOG(LEVEL) << "prepareScintillatorLib" ;  
    prepareScintillatorLib();

    LOG(LEVEL) << "prepareSourceLib" ;  
    prepareSourceLib();

    LOG(LEVEL) << "prepareVolumes" ;  
    prepareVolumes();   // GInstancer::createInstancedMergedMeshes

    LOG(LEVEL) << "prepareVertexColors" ;  
    prepareVertexColors();  // writes colors into GMergedMesh mm0

    LOG(info) << "]" ; 
}





void GGeo::save()
{
    const char* idpath = m_ok->getIdPath() ;
    assert( idpath ); 
    LOG(LEVEL) << "[" << " idpath " << idpath ; 

    if(!m_prepared)
    {
        LOG(info) << "preparing before save " ; 
        prepare();
    }   

    m_geolib->dump("GGeo::save");

    m_geolib->save(); // in here GGeoLib::saveConstituents invokes the save of both triangulated GMergedMesh and analytic GParts 
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


void GGeo::saveCacheMeta() const 
{
    if(m_gdmlauxmeta)
    {
         const char* gdmlauxmetapath = m_ok->getGDMLAuxMetaPath(); 
         m_gdmlauxmeta->save(gdmlauxmetapath); 
    }

    if(m_lv2sd)
    {
        m_ok->appendCacheMeta("lv2sd", m_lv2sd); 
    } 
    if(m_lv2mt)
    {
        m_ok->appendCacheMeta("lv2mt", m_lv2mt); 
    } 


    m_ok->dumpCacheMeta("GGeo::saveCacheMeta"); 
    m_ok->saveCacheMeta(); 
}


void GGeo::saveGLTF() const 
{
    int root = SSys::getenvint( "GLTF_ROOT", 3147 );  
    const char* gltfpath = m_ok->getGLTFPath(); 
    m_ok->profile("_GGeo::saveGLTF"); 
    GGeoGLTF::Save(this, gltfpath, root );  
    m_ok->profile("GGeo:saveGLTF"); 
}




/**
GGeo::loadCacheMeta
----------------------

Invoked at the tail of GGeo::loadFromCache.

* see Opticks::loadOriginCacheMeta

**/

void GGeo::loadCacheMeta() // loads metadata that the process that created the geocache persisted into the geocache
{
    LOG(LEVEL) ; 

    NMeta* gdmlauxmeta = m_ok->getGDMLAuxMeta(); 
    setGDMLAuxMeta(gdmlauxmeta); 

    NMeta* lv2sd = m_ok->getOriginCacheMeta("lv2sd"); 
    NMeta* lv2mt = m_ok->getOriginCacheMeta("lv2mt"); 

    //if(lv2sd) lv2sd->dump("GGeo::loadCacheMeta.lv2sd"); 
    //if(lv2mt) lv2mt->dump("GGeo::loadCacheMeta.lv2mt"); 

    if( m_ok->isTest() )   // --test : skip lv2sd association
    {
        LOG(LEVEL) << "NOT USING the lv2sd lv2mt association as --test is active " ;  
    }
    else
    {
        m_lv2sd = lv2sd ;  
        m_lv2mt = lv2mt ;  
    }
}








bool GGeo::isLive() const 
{
    return m_live ; 
}

bool GGeo::isValid() const 
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


void GGeo::Summary(const char* msg)
{
    LOG(info) 
        << msg
        << " ms " << m_meshlib->getNumMeshes()
        << " so " << m_nodelib->getNumVolumes()
        << " mt " << m_materiallib->getNumMaterials()
        << " bs " << getNumBorderSurfaces() 
        << " ss " << getNumSkinSurfaces()
        ;

}

void GGeo::Details(const char* msg)
{
    Summary(msg) ;

    char mbuf[BSIZ];

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

}








//  via GMeshLib

GMeshLib* GGeo::getMeshLib()
{
    return m_meshlib ; 
}
unsigned GGeo::getNumMeshes() const 
{
    return m_meshlib->getNumMeshes(); 
}

const GMesh* GGeo::getMesh(unsigned aindex) const 
{
    return m_meshlib->getMeshWithIndex(aindex);
}  
void GGeo::add(const GMesh* mesh)  // canonically invoked by X4PhysicalVolume::convertSolids_r
{
    m_meshlib->add(mesh);
}
void GGeo::countMeshUsage(unsigned meshIndex, unsigned nodeIndex)
{
    m_meshlib->countMeshUsage(meshIndex, nodeIndex); 
}
void GGeo::reportMeshUsage(const char* msg) const 
{
    m_meshlib->reportMeshUsage(msg);
}
 

/**
GGeo::setRootVolume (via GNodeLib)
------------------------------------

Canonically invoked by X4PhysicalVolume::convertStructure, 

**/
void GGeo::setRootVolume(const GVolume* root)
{
    m_nodelib->setRootVolume(root); 
}
const GVolume* GGeo::getRootVolume() const 
{
    return m_nodelib->getRootVolume(); 
}

unsigned GGeo::getNumVolumes() const 
{
    return m_nodelib->getNumVolumes();
}
void GGeo::addVolume(const GVolume* volume)
{
    m_nodelib->addVolume(volume);
}


NPY<float>* GGeo::getTransforms() const 
{
    return m_nodelib->getTransforms(); 
}
NPY<float>* GGeo::getInverseTransforms() const 
{
    return m_nodelib->getInverseTransforms(); 
}


const GVolume* GGeo::getVolume(unsigned index) const 
{
    return m_nodelib->getVolume(index);
}
const GVolume* GGeo::getVolumeSimple(unsigned int index) const 
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
const GNode* GGeo::getNode(unsigned index) const 
{
    return m_nodelib->getNode(index);
}

unsigned GGeo::getNumSensorVolumes() const 
{
    return m_nodelib->getNumSensorVolumes() ; 
}
glm::uvec4 GGeo::getSensorIdentity(unsigned sensorIndex) const 
{
    return m_nodelib->getSensorIdentity(sensorIndex); 
}
unsigned GGeo::getSensorIdentityStandin(unsigned sensorIndex) const 
{
    return m_nodelib->getSensorIdentityStandin(sensorIndex); 
}

const GVolume* GGeo::getSensorVolume(unsigned sensorIndex) const 
{
    return m_nodelib->getSensorVolume(sensorIndex) ; 
}
std::string GGeo::reportSensorVolumes(const char* msg) const 
{
    return m_nodelib->reportSensorVolumes(msg); 
}
void GGeo::dumpSensorVolumes(const char* msg) const 
{
    m_nodelib->dumpSensorVolumes(msg); 
}
void GGeo::getSensorPlacements(std::vector<void*>& placements, bool outer_volume) const 
{
    m_nodelib->getSensorPlacements(placements, outer_volume); 
}


int GGeo::getFirstNodeIndexForGDMLAuxTargetLVName() const 
{
    return m_nodelib->getFirstNodeIndexForGDMLAuxTargetLVName() ; 
}

void GGeo::getNodeIndicesForLVName(std::vector<unsigned>& nidxs, const char* lvname) const 
{
    m_nodelib->getNodeIndicesForLVName(nidxs, lvname); 
}



void GGeo::dumpNodes(const std::vector<unsigned>& nidxs, const char* msg) const 
{
    m_nodelib->dumpNodes(nidxs,msg);
}


// via GMaterialLib


GMaterial* GGeo::getMaterial(unsigned aindex) const 
{
    return m_materiallib->getMaterialWithIndex(aindex);
}
GPropertyMap<float>* GGeo::findMaterial(const char* shortname) const 
{
    return m_materiallib->findMaterial(shortname) ; 
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


// via GSurfaceLib

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


void GGeo::traverse( const GNode* node, unsigned depth)
{
    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1);
}





void GGeo::prepareMaterialLib()
{
    LOG(verbose) ;

    GMaterialLib* mlib = getMaterialLib() ;
   
    mlib->addTestMaterials(); 
}

void GGeo::prepareSurfaceLib()
{
    LOG(verbose) ; 

    GSurfaceLib* slib = getSurfaceLib() ;
   
    slib->addPerfectSurfaces(); 
}


void GGeo::prepareSourceLib()
{
    LOG(verbose) ;

    GSourceLib* srclib = getSourceLib() ;

    srclib->close();
}



/**
GGeo::close
-------------

This needs to be invoked after all Opticks materials and surfaces have been
created, and before boundaries are formed : typically in the recursive structure traverse


**/

void GGeo::close()
{
    LOG(LEVEL) << "[" ; 

    GMaterialLib* mlib = getMaterialLib() ;
    GSurfaceLib* slib = getSurfaceLib() ;

    mlib->close();
    slib->close();

    // this was not here traditionally due to late addition of boundaries 
    GBndLib* blib = getBndLib() ;
    blib->createDynamicBuffers(); 

    LOG(LEVEL) << "]" ; 
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

This is invoked by GGeo::prepare. 

This uses GInstancer to create the instanced GMergedMesh by combinations
of volumes from the GVolume tree. 
The created GMergedMesh are collected into GGeo/GGeoLib.

As this is creating GMergedMesh it is clearly precache.
(Now that GMeshLib are persisted this is not so clearly precache, 
but that remains the case typically.)

**/

void GGeo::prepareVolumes()
{
    LOG(info) << "[ creating merged meshes from the volume tree " ; 

    assert( m_instancer && "prepareVolumes can only be done pre-cache when the full node tree is available"); 

    unsigned numcsgskiplv = m_ok->getNumCSGSkipLV() ; 
    if(numcsgskiplv > 0)
    {
        LOG(fatal) << " numcsgskiplv " << numcsgskiplv ; 
    }

    bool instanced = m_ok->isInstanced();
    unsigned meshverbosity = m_ok->getMeshVerbosity() ; 

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
        LOG(fatal) << "instancing inhibited " ;
        const GNode* root = getNode(0);
        m_geolib->makeMergedMesh(0, NULL, root );  // ridx:0 rbase:NULL 
    }

    m_instancer->dump("GGeo::prepareVolumes") ; 
    LOG(info) << "]" ;
}


/**
GGeo::deferredCreateGParts
------------------------------

This is invoked from on high in OpticksHub::init/OpticksHub::deferredGeometryPrep
after GGeo geometry is loaded or adopted into OpticksHub.

This is needed prior to GPU upload of analytic geometry by OGeo,
it requires the GMergedMesh from GGeoLib and the NCSG solids from GMeshLib.  
Thus it can be done postcache, as all the ingredients are loaded from cache.

See notes/issues/GPts_GParts_optimization.rst

**/

void GGeo::deferredCreateGParts()
{
    LOG(LEVEL) << "[" ; 

    const std::vector<const NCSG*>& solids = m_meshlib->getSolids(); 
          
    //unsigned verbosity = 0 ;  

    unsigned nmm = m_geolib->getNumMergedMesh(); 

    LOG(LEVEL) 
        << " geolib.nmm " << nmm 
        << " meshlib.solids " << solids.size()
        ; 

    for(unsigned i=0 ; i < nmm ; i++)
    {
        GMergedMesh* mm = m_geolib->getMergedMesh(i);

        if( mm->getParts() != NULL )
        {
            LOG(LEVEL) << " skip as parts already present for mm " << i ;  
            // this happens for test geometry eg tboolean.sh 
            continue ; 
        } 

        assert( mm->getParts() == NULL ); 

        GPts* pts = mm->getPts(); 
        if( pts == NULL )
        { 
            LOG(fatal) << " pts NULL, cannot create GParts for mm " << i ; 
            //continue ; 
            assert(0); 
        }

        GParts* parts = GParts::Create( pts, solids) ; 
        parts->setBndLib(m_bndlib); 
        parts->close(); 

        mm->setParts( parts ); 
    }

    LOG(LEVEL) << "]" ; 
}





GMergedMesh* GGeo::makeMergedMesh(unsigned int index, const GNode* base, const GNode* root)
{
    GGeoLib* geolib = getGeoLib() ;
    assert(geolib);
    return geolib->makeMergedMesh(index, base, root);
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
    LOG(info) << msg ; 

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






unsigned GGeo::getNumRepeats() const 
{
    return m_geolib->getNumRepeats(); 
}
unsigned GGeo::getNumPlacements(unsigned ridx) const 
{
    return m_geolib->getNumPlacements(ridx); 
}
unsigned GGeo::getNumVolumes(unsigned ridx) const 
{
    return m_geolib->getNumVolumes(ridx); 
}


void GGeo::dumpNode(unsigned nidx)
{
    glm::uvec4 id = m_nodelib->getIdentity(nidx); 
    unsigned triplet = id.y ;
    unsigned ridx = OpticksIdentity::RepeatIndex(triplet); 
    unsigned pidx = OpticksIdentity::PlacementIndex(triplet); 
    unsigned oidx = OpticksIdentity::OffsetIndex(triplet); 
    dumpNode(ridx, pidx, oidx); 
}
void GGeo::dumpNode(unsigned ridx, unsigned pidx, unsigned oidx)
{
    glm::uvec4 id = m_geolib->getIdentity(ridx, pidx, oidx); 
    unsigned nidx = id.x ; 
    LOG(info)
        << " ("
        << " ridx " << ridx
        << " pidx " << pidx
        << " oidx " << oidx
        << " )"
        << " nidx " << nidx
        ;
}

glm::uvec4 GGeo::getIdentity(unsigned nidx) const 
{
    glm::uvec4 id = m_nodelib->getIdentity(nidx); 
    return id ; 
}
glm::uvec4 GGeo::getNRPO(unsigned nidx) const 
{
    glm::uvec4 nrpo = m_nodelib->getNRPO(nidx); 
    return nrpo ; 
}

glm::uvec4 GGeo::getIdentity(unsigned ridx, unsigned pidx, unsigned oidx, bool check) const 
{
    glm::uvec4 id = m_geolib->getIdentity(ridx, pidx, oidx); 
    if(check)
    {
        // consistency check the identity with that obtained from nodelib
        unsigned nidx = id.x ; 
        glm::uvec4 id2 = m_nodelib->getIdentity(nidx);  
        assert( id.x == id2.x );  
        assert( id.y == id2.y );  
        assert( id.z == id2.z );  
        assert( id.w == id2.w );  

        // consistency check the triplet identity  
        unsigned triplet = id.y ;
        assert( OpticksIdentity::RepeatIndex(triplet)    == ridx ); 
        assert( OpticksIdentity::PlacementIndex(triplet) == pidx ); 
        assert( OpticksIdentity::OffsetIndex(triplet)    == oidx ); 
    }
    return id ; 
}

glm::mat4 GGeo::getTransform(unsigned ridx, unsigned pidx, unsigned oidx, bool check) const 
{
    glm::mat4  tr = m_geolib->getTransform(ridx, pidx, oidx) ;

    if(check)
    {
        glm::uvec4 id = m_geolib->getIdentity(ridx, pidx, oidx) ;
        unsigned nidx = id.x ; 
        // consistency check the nodelib transform
        glm::mat4 tr2 = m_nodelib->getTransform(nidx); 

        float epsilon = 1e-5 ; 
        float diff = nglmext::compDiff(tr, tr2 ); 

        LOG(LEVEL) 
            << " ridx " << ridx
            << " pidx " << pidx
            << " oidx " << oidx
            << OpticksIdentity::Desc(" id",id) 
            << " diff*1.e9 " << diff*1.0e9 
            << " epsilon " << epsilon 
            ;

        bool expected = std::abs(diff) < epsilon ; 
        if(!expected)
        {
            LOG(fatal)
               << " epsilon " << epsilon
               << " diff " << diff
               ;

            std::cout << gpresent("tr", tr) << std::endl ;
            std::cout << gpresent("tr2", tr2) << std::endl ;
        }
        assert( expected );
    }

    return tr ; 
}





void GGeo::dumpShape(const char* msg) const 
{
    const GGeo* gg = this ; 
    unsigned num_repeats = gg->getNumRepeats(); 
    LOG(info) << msg << " num_repeats " << num_repeats ; 
    for(unsigned ridx=0 ; ridx < num_repeats ; ridx++)
    {
         unsigned num_placements = gg->getNumPlacements(ridx); 
         unsigned num_volumes = gg->getNumVolumes(ridx); 
         std::cout 
             << " ridx " << std::setw(3) << ridx
             << " num_placements  " << std::setw(6) << num_placements
             << " num_volumes  " << std::setw(6) << num_volumes
             << std::endl 
             ;     
    }
}





unsigned GGeo::getNumTransforms() const 
{
    return m_nodelib->getNumTransforms(); 
}
glm::mat4 GGeo::getTransform(unsigned index) const 
{
    return m_nodelib->getTransform(index); 
}
glm::mat4 GGeo::getInverseTransform(unsigned index) const 
{
    return m_nodelib->getInverseTransform(index); 
}




void GGeo::dumpVolumes(const std::map<std::string, int>& targets, const char* msg, float extent_cut_mm, int cursor ) const 
{
    m_nodelib->dumpVolumes(targets, msg, extent_cut_mm, cursor); 
}
glm::vec4 GGeo::getCE(unsigned index) const 
{
    return m_nodelib->getCE(index); 
}


/**
GGeo::dumpTree
---------------

This formerly dumped all volumes from mm0. 
Following the model change this wil dump just the 
remainder volumes.

**/

void GGeo::dumpTree(const char* msg)
{
    GMergedMesh* mm0 = getMergedMesh(0);

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

         const char* pv = m_nodelib->getPVName(i);
         const char* lv = m_nodelib->getLVName(i);

         printf(" %6u : nf %4d nv %4d id %6u pid %6d : %4d %4d %4d %4d  :%50s %50s \n", i, 
                    info->x, info->y, info->z, info->w,  offnum.x, offnum.y, offnum.z, offnum.w,
                    pv, lv ); 
    }
}




/**
GGeo::getNodeOffsetCount
-------------------------

Adds face and vertex counts for all volumes up to volume i, 
giving offsets into the merged arrays. 

Hmm: need to migrate nodeinfo to GNodeLib too ?

**/

glm::ivec4 GGeo::getNodeOffsetCount(unsigned index) // TODO: move into geolib
{
    GMergedMesh* mm0 = getMergedMesh(0);

    guint4* nodeinfo = mm0->getNodeInfo(); 
    unsigned num_vol = mm0->getNumVolumes();  
    assert(index < num_vol );

    glm::ivec4 offset ; 
    unsigned cur_vert(0);
    unsigned cur_face(0);

    for(unsigned i=0 ; i < num_vol ; i++)
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
    LOG(LEVEL)
        << " evt " << evt 
        ;
}


/**
GGeo::dryrun_convert
-----------------------

This is a dryrun of OGeo::convert with no GPU involvement for safety.
Passing the dryrun is advisable prior to trying OGeo::convert onto
the GPU as uploading a broken geometry to the GPU tends to cause 
hard crashes and kernel panics.

**/

void GGeo::dryrun_convert() 
{
    deferredCreateGParts();  // hmm : find somewhere better to do this 
    m_geolib->dryrun_convert(); 
}



