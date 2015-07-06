#include "GLoader.hh"

// look no hands : no dependency on AssimpWrap 

#include "GMergedMesh.hh"
#include "GBoundaryLib.hh"
#include "GSensorList.hh"
#include "GBoundaryLibMetadata.hh"
#include "GTraverse.hh"

//#include "GMaterialIndex.hh"
//#include "GSurfaceIndex.hh"
//#include "GFlagIndex.hh"
#include "GItemIndex.hh"

#include "GGeo.hh"
#include "GCache.hh"
#include "GColors.hh"
#include "GColorMap.hh"

// npy-
#include "stringutil.hpp"
#include "Lookup.hpp"
#include "Types.hpp"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void GLoader::load(bool nogeocache)
{
    assert(m_cache);
    const char* idpath = m_cache->getIdPath() ;
    const char* envprefix = m_cache->getEnvPrefix() ;

    LOG(info) << "GLoader::load start idpath " << idpath << " nogeocache " << nogeocache  ;

    fs::path geocache(idpath);

    if(fs::exists(geocache) && fs::is_directory(geocache) && !nogeocache ) 
    {
        LOG(info) << "GLoader::load loading from cache directory " << idpath ;
        m_ggeo = NULL ; 
        m_mergedmesh = GMergedMesh::load(idpath);
        m_metadata   = GBoundaryLibMetadata::load(idpath);
        m_materials  = GItemIndex::load(idpath, "GMaterialIndex"); // TODO: find common place for such strings, maybe Types.hpp
        m_surfaces   = GItemIndex::load(idpath, "GSurfaceIndex");
    } 
    else
    {
        LOG(info) << "GLoader::load slow loading using m_imp (disguised AssimpGGeo) " << envprefix ;
        m_ggeo = (*m_imp)(envprefix);      

        GBoundaryLib* lib = m_ggeo->getBoundaryLib();

        lib->getMaterials()->loadIndex("$HOME/.opticks"); // customize GMaterialIndex

        m_ggeo->sensitize(idpath, "idmap");  // loads idmap and traverses nodes doing GSolid::setSensor for sensitve nodes

        m_mergedmesh = m_ggeo->getMergedMesh();  // creates merged mesh, doing the flattening  
        m_mergedmesh->setColor(0.5,0.5,1.0);
        LOG(info) << "GLoader::load saving to cache directory " << idpath ;
        m_mergedmesh->save(idpath); 

        m_metadata = lib->getMetadata();
        m_metadata->save(idpath);

        m_materials = lib->getMaterials();  
        m_materials->save(idpath);
        
        m_surfaces = lib->getSurfaces();   
        m_surfaces->save(idpath);


        lib->saveIndex(idpath); 
    } 
  
    // hmm not routing via cache 
    m_lookup = new Lookup() ; 
    m_lookup->create(idpath);

    Index* idx = m_types->getFlagsIndex() ;    
    m_flags = new GItemIndex( idx );     //GFlagIndex::load(idpath); 

    // itemname => colorname 
    m_materials->setColorMap(GColorMap::load(idpath, "GMaterialIndexColors.json")); 
    m_surfaces->setColorMap(GColorMap::load(idpath, "GSurfaceIndexColors.json"));   
    m_flags->setColorMap(GColorMap::load(idpath, "GFlagIndexColors.json"));    

    m_colors = GColors::load(idpath,"GColors.json");                         // colorname => hexcode 

    m_materials->setLabeller(GItemIndex::COLORKEY);
    m_materials->setColorSource(m_colors);
    m_surfaces->setColorSource(m_colors);
    m_flags->setColorSource(m_colors);

    m_flags->formTable(); // as not yet pulling a buffer

    GBuffer* buffer = m_materials->makeColorBuffer();
    //m_colors->dump_uchar4_buffer(buffer);

    LOG(info) << "GLoader::load done " << idpath ;
    assert(m_mergedmesh);
}

void GLoader::Summary(const char* msg)
{
    printf("%s\n", msg);
    m_mergedmesh->Summary("GLoader::Summary");
    m_mergedmesh->Dump("GLoader::Summary Dump",10);
}



