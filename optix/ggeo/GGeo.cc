#include "GGeo.hh"

#include "GCache.hh"
#include "GSkinSurface.hh"
#include "GBorderSurface.hh"
#include "GMaterial.hh"
#include "GPropertyMap.hh"
#include "GSolid.hh"
#include "GMesh.hh"
#include "GTreeCheck.hh"
#include "GTreePresent.hh"
#include "GColorizer.hh"
#include "GGeoTestConfig.hh"
#include "GGeoTest.hh"
#include "GPmt.hh"
#include "GColors.hh"

#include "GGeoLib.hh"
#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GScintillatorLib.hh"
#include "GSourceLib.hh"
#include "GFlags.hh"
#include "GAttrSeq.hh"

#include "GMergedMesh.hh"
#include "GColors.hh"
#include "GItemIndex.hh"
#include "GItemList.hh"


#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "TorchStepNPY.hpp"
#include "NSensorList.hpp"
#include "NSensor.hpp"
#include "Lookup.hpp"
#include "Typ.hpp"


// opticks-
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "Composition.hh"

#include "assert.h"
#include "stdio.h"
#include "string.h"
#include "stringutil.hpp"

#include <iomanip>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "NLog.hpp"


#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#define BSIZ 50

const char* GGeo::CATHODE_MATERIAL = "Bialkali" ; 
const char* GGeo::PICKFACE = "pickface" ;


void GGeo::init()
{
   m_cache->setGGeo(this); 

   Opticks* opticks = m_cache->getOpticks(); 
   OpticksResource* resource = m_cache->getResource(); 

   const char* idpath = resource->getIdPath() ;

   fs::path geocache(idpath); 

   bool gc_exists = fs::exists(geocache) && fs::is_directory(geocache) ;
   bool gc_request = opticks->isGeocache() ; 

   m_loaded = gc_exists && gc_request ;

   LOG(debug) << "GGeo::init"
             << " idpath " << idpath
             << " gc_exists " << gc_exists 
             << " gc_request " << gc_request
             << " m_loaded " << m_loaded 
             ;

   const char* ctrl = resource->getCtrl() ;

   m_volnames = GGeo::ctrlHasKey(ctrl, "volnames");
 
   m_sensor_list = new NSensorList();

   m_sensor_list->load( idpath, "idmap");


   LOG(debug) << "GGeo::init loadSensorList " << m_sensor_list->description() ; 

   if(m_loaded) return ; 

   //////////////  below only when operating pre-cache //////////////////////////

   m_geolib = new GGeoLib(m_cache);

   m_treecheck = new GTreeCheck(this) ;
   if(resource->isJuno())
       m_treecheck->setVertexMin(250);

   m_treepresent = new GTreePresent(this, 0, 100, 1000);   // top,depth_max,sibling_max

   //GColorizer::Style_t style  = GColorizer::SURFACE_INDEX ;  // rather grey 
   GColorizer::Style_t style = GColorizer::PSYCHEDELIC_NODE ;

   m_colorizer = new GColorizer( this, style ); // colorizer needs full tree, so pre-cache only 

   m_bndlib = new GBndLib(m_cache);
   m_materiallib = new GMaterialLib(m_cache);
   m_surfacelib  = new GSurfaceLib(m_cache);

   m_bndlib->setMaterialLib(m_materiallib);
   m_bndlib->setSurfaceLib(m_surfacelib);

   m_scintillatorlib  = new GScintillatorLib(m_cache);
   m_sourcelib  = new GSourceLib(m_cache);

   m_meshindex = new GItemIndex("MeshIndex") ; 

   if(m_volnames)
   {
       m_pvlist = new GItemList("PVNames") ; 
       m_lvlist = new GItemList("LVNames") ; 
   }
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
    m_surfacelib->add(surface);
    m_skin_surfaces.push_back(surface);
    addToIndex((GPropertyMap<float>*)surface);
}



unsigned int GGeo::getNumMergedMesh()
{
    return m_geolib->getNumMergedMesh();
}

GMergedMesh* GGeo::getMergedMesh(unsigned int index)
{
    return m_geolib->getMergedMesh(index);
}

GMergedMesh* GGeo::makeMergedMesh(unsigned int index, GNode* base)
{
    return m_geolib->makeMergedMesh(this, index, base);
}



const char* GGeo::getIdPath()
{
    return m_cache->getIdPath();
}
GColors* GGeo::getColors()
{
   return m_cache->getColors() ; 
}
GFlags* GGeo::getFlags()
{
    return m_cache->getFlags();
}

unsigned int GGeo::getMaterialLine(const char* shortname)
{
    return m_bndlib->getMaterialLine(shortname);
}

void GGeo::loadGeometry()
{
    LOG(debug) << "GGeo::loadGeometry START" ; 
    const char* idpath = getIdPath() ;

    if(!isLoaded())
    {
        loadFromG4DAE();
        save(idpath);
    }
    else
    {
        loadFromCache();
    } 

    loadAnalyticPmt();

    setupLookup();
    setupColors();
    setupTyp();
    LOG(debug) << "GGeo::loadGeometry DONE" ; 
}

void GGeo::loadFromG4DAE()
{
    LOG(info) << "GGeo::loadFromG4DAE START" ; 

    int rc = (*m_loader_imp)(this);   //  imp set in main: m_ggeo->setLoaderImp(&AssimpGGeo::load); 

    assert(rc == 0);

    prepareScintillatorLib();

    prepareMeshes();

    prepareVertexColors();

    LOG(info) << "GGeo::loadFromG4DAE DONE" ; 
}

void GGeo::afterConvertMaterials()
{
    LOG(debug) << "GGeo::afterConvertMaterials and before convertStructure" ; 

    prepareMaterialLib(); 
    prepareSurfaceLib(); 
}


bool GGeo::isValid()
{
    return m_bndlib->isValid() && m_materiallib->isValid() && m_surfacelib->isValid() ; 
}

void GGeo::loadFromCache()
{   
    LOG(debug) << "GGeo::loadFromCache START" ; 

    m_geolib = GGeoLib::load(m_cache);
        
    const char* idpath = m_cache->getIdPath() ;
    m_meshindex = GItemIndex::load(idpath, "MeshIndex");

    if(m_volnames)
    {
        m_pvlist = GItemList::load(idpath, "PVNames");
        m_lvlist = GItemList::load(idpath, "LVNames");
    }

    m_bndlib = GBndLib::load(m_cache);  // GBndLib is persisted via index buffer, not float buffer

    

    m_materiallib = GMaterialLib::load(m_cache);
    m_surfacelib  = GSurfaceLib::load(m_cache);
    m_bndlib->setMaterialLib(m_materiallib);
    m_bndlib->setSurfaceLib(m_surfacelib);

    m_scintillatorlib  = GScintillatorLib::load(m_cache);
    m_sourcelib  = GSourceLib::load(m_cache);



    LOG(debug) << "GGeo::loadFromCache DONE" ; 
}



void GGeo::setupLookup()
{
    // see ggeo-/tests/LookupTest.cc
    m_lookup = new Lookup() ; 

    m_lookup->loadA( m_cache->getResource()->getIdFold(), "ChromaMaterialMap.json", "/dd/Materials/") ;

    m_bndlib->fillMaterialLineMap( m_lookup->getB() ) ;    

    m_lookup->crossReference();

    //m_lookup->dump("GGeo::setupLookup");  
}

void GGeo::setupTyp()
{
   // hmm maybe better elsewhere to avoid repetition from tests ? 
    Typ* typ = m_cache->getTyp();
    GFlags* flags = m_cache->getFlags();
    typ->setMaterialNames(m_materiallib->getNamesMap());
    typ->setFlagNames(flags->getNamesMap());
}

void GGeo::setupColors()
{
    LOG(debug) << "GGeo::setupColors" ; 

    GFlags* flags = m_cache->getFlags();

    std::vector<unsigned int>& material_codes = m_materiallib->getAttrNames()->getColorCodes() ; 
    std::vector<unsigned int>& flag_codes     = flags->getAttrIndex()->getColorCodes() ; 

    GColors* colors = m_cache->getColors();
    colors->setupCompositeColorBuffer( material_codes, flag_codes  );
}

void GGeo::save(const char* idpath)
{
    m_geolib->saveToCache();

    m_meshindex->save(idpath);

    if(m_volnames)
    {
        m_pvlist->save(idpath);
        m_lvlist->save(idpath);
    }

    // details of save handled within the class, not here 
    m_materiallib->save();
    m_surfacelib->save();
    m_scintillatorlib->save();
    m_sourcelib->save();
    m_bndlib->save();  
}


void GGeo::loadAnalyticPmt()
{
    NSlice* slice = NULL ;

    unsigned int pmtIndex = 0 ;  

    m_pmt = GPmt::load( m_cache, m_bndlib, pmtIndex, slice ); 

    if(m_pmt)
    {
        LOG(info) << "GGeo::loadAnalyticPmt SUCCEEDED"
                  << m_pmt->getPath()   
                   ;
    }

}




void GGeo::modifyGeometry(const char* config)
{
    // NB only invoked with test option : "ggv --test" 
    GGeoTestConfig* gtc = new GGeoTestConfig(config);

    LOG(debug) << "GGeo::modifyGeometry" 
              << " config [" << ( config ? config : "" ) << "]" ; 

    assert(m_geotest == NULL);

    m_geotest = new GGeoTest(m_cache, gtc);
    m_geotest->modifyGeometry();
}






const char* GGeo::getPVName(unsigned int index)
{
    return m_pvlist ? m_pvlist->getKey(index) : NULL ; 
}
const char* GGeo::getLVName(unsigned int index)
{
    return m_lvlist ? m_lvlist->getKey(index) : NULL ; 
}

bool GGeo::ctrlHasKey(const char* ctrl, const char* key)
{
    return listHasKey(ctrl, key, ",");
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
    printf("%s ms %lu so %lu mt %lu bs %lu ss %lu \n", msg, m_meshes.size(), m_solids.size(), m_materials.size(), m_border_surfaces.size(), m_skin_surfaces.size() );  

    if(m_low)  printf("    low  %10.3f %10.3f %10.3f \n", m_low->x, m_low->y, m_low->z);
    if(m_high) printf("    high %10.3f %10.3f %10.3f \n", m_high->x, m_high->y, m_high->z);
}

void GGeo::Details(const char* msg)
{
    printf("%s  #border_surfaces %lu #skin_surfaces %lu #materials %lu \n", msg, m_border_surfaces.size(),  m_skin_surfaces.size(), m_materials.size()); 
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





GMesh* GGeo::getMesh(unsigned int aindex)
{
    GMesh* mesh = NULL ; 
    for(unsigned int i=0 ; i < m_meshes.size() ; i++ )
    { 
        if(m_meshes[i]->getIndex() == aindex )
        {
            mesh = m_meshes[i] ; 
            break ; 
        }
    }
    return mesh ;
}  

/*

hmm the individual mesh aint going to be there post-cache 

unsigned int GGeo::getNextMeshIndex()
{
    unsigned int mn(0) ;
    unsigned int mx(0) ;
    GMesh* mesh = NULL ; 
    for(unsigned int i=0 ; i < m_meshes.size() ; i++ )
    { 
        mesh = 

    }
}
*/



void GGeo::add(GMesh* mesh)
{
    m_meshes.push_back(mesh);

    const char* name = mesh->getName();
    unsigned int index = mesh->getIndex();

    LOG(debug) << "GGeo::add (GMesh)"
              << " index " << std::setw(4) << index 
              << " name " << name 
              ;

    m_meshindex->add(name, index); 
}

void GGeo::add(GSolid* solid)
{
    m_solids.push_back(solid);
    unsigned int index = solid->getIndex(); // absolute node index, independent of the selection
    //printf("GGeo::add solid %u \n", index);
    m_solidmap[index] = solid ; 

    if(m_volnames)
    { 
        m_lvlist->add(solid->getLVName()); 
        m_pvlist->add(solid->getPVName()); 
    }

    GSolid* check = getSolid(index);
    assert(check == solid);
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


GSolid* GGeo::getSolid(unsigned int index)
{
    GSolid* solid = NULL ; 
    if(m_solidmap.find(index) != m_solidmap.end()) 
    {
        solid = m_solidmap[index] ;
        assert(solid->getIndex() == index);
    }
    return solid ; 
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
    LOG(info) << "GGeo::prepareMaterialLib " ; 

    GMaterialLib* mlib = getMaterialLib() ;
   
    mlib->addTestMaterials(); 
}

void GGeo::prepareSurfaceLib()
{
    LOG(info) << "GGeo::prepareSurfaceLib " ; 

    GSurfaceLib* slib = getSurfaceLib() ;
   
    slib->addPerfectSurfaces(); 
}



void GGeo::prepareScintillatorLib()
{
    LOG(info) << "GGeo::prepareScintillatorLib " ; 

    findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB"); 

    unsigned int nscint = getNumScintillatorMaterials() ;

    if(nscint == 0)
    {
        LOG(warning) << "GGeo::prepareScintillatorLib found no scintillator materials  " ; 
    }
    else
    {
        GPropertyMap<float>* scint = dynamic_cast<GPropertyMap<float>*>(getScintillatorMaterial(0));  

        GScintillatorLib* sclib = getScintillatorLib() ;

        sclib->add(scint);

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
    bool instanced = m_cache->getOpticks()->isInstanced();

    LOG(info) << "GGeo::prepareMeshes START" 
              << " instanced " << instanced 
              ;

    if(instanced)
    { 
        bool deltacheck = true ; 
        m_treecheck->createInstancedMergedMeshes(deltacheck);   // GTreeCheck::createInstancedMergedMeshes
    }
    else
    {
        LOG(warning) << "GGeo::prepareMeshes instancing inhibited " ;
        makeMergedMesh(0, NULL);  // ridx:0 rbase:NULL 
    }
    LOG(info) << "GGeo::prepareMeshes DONE" ;
}


void GGeo::prepareVertexColors()
{
    // GColorizer needs full tree,  so have to use pre-cache
    GMergedMesh* mesh0 = getMergedMesh(0);
    gfloat3* vertex_colors = mesh0->getColors();

    GColorizer* czr = getColorizer();

    czr->setTarget( vertex_colors );
    //czr->setSurfaces(m_surfaces);   NO LONGER USING this GLoader approach 
    czr->setRepeatIndex(mesh0->getIndex()); 
    czr->traverse();

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





void GGeo::countMeshUsage(unsigned int meshIndex, unsigned int nodeIndex, const char* lv, const char* pv)
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



void GGeo::targetTorchStep( TorchStepNPY* torchstep )
{
    // targetted positioning and directioning of the torch requires geometry info, 
    // which is not available within npy- so need to externally setFrameTransform
    // based on integer frame volume index

    if(torchstep->isFrameTargetted())
    { 
        LOG(info) << "GGeo::targetTorchStep frame targetted already  " << gformat(torchstep->getFrameTransform()) ;  
    }
    else
    {
        glm::ivec4& iframe = torchstep->getFrame();
        glm::mat4 transform = getTransform( iframe.x );
        LOG(debug) << "GGeo::targetTorchStep setting frame " << iframe.x << " " << gformat(transform) ;  
        torchstep->setFrameTransform(transform);
    }

    //glm::vec3 pol( 0.f, 0.f, 1.f);  // currently ignored
    //torchstep->setPolarization(pol);

}



void GGeo::dumpTree(const char* msg)
{
    GMergedMesh* mm0 = getMergedMesh(0);

    // all these are full traverse counts, not reduced by selections or instancing
    unsigned int nso = mm0->getNumSolids();  
    guint4* nodeinfo = mm0->getNodeInfo(); 
    unsigned int npv = m_pvlist->getNumKeys(); 
    unsigned int nlv = m_lvlist->getNumKeys(); 

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
         const char* pv = m_pvlist->getKey(i);
         const char* lv = m_lvlist->getKey(i);
         printf(" %6u : nf %4d nv %4d id %6u pid %6d : %4d %4d %4d %4d  :%50s %50s \n", i, 
                    info->x, info->y, info->z, info->w,  offnum.x, offnum.y, offnum.z, offnum.w,
                    pv, lv ); 
    }
}


glm::ivec4 GGeo::getNodeOffsetCount(unsigned int index)
{
    GMergedMesh* mm0 = getMergedMesh(0);
    guint4* nodeinfo = mm0->getNodeInfo(); 
    unsigned int nso = mm0->getNumSolids();  
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

    for(unsigned int i=0 ; i < offnum.z ; i++)
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

    assert(face_index0 <  offnum.z );  
    assert(face_index1 <= offnum.z );   // face_index1 needs to go 1 beyond

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
    bool join = m_join_cfg && listHasKey(m_join_cfg, shortname, ",") ; 

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
        result = (*m_join_imp)(mesh, m_cache ); 

        result->setName(mesh->getName()); 
        result->setIndex(mesh->getIndex()); 
        result->updateBounds();
    }
    return result ; 
}







// pickface machinery must be here as GGeo cannot live in Opticks

void GGeo::setPickFace(std::string pickface)
{
    setPickFace(givec4(pickface));
}

glm::ivec4& GGeo::getPickFace()
{
    return m_composition->getPickFace();
}

void GGeo::setPickFace(glm::ivec4 pickface) 
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







