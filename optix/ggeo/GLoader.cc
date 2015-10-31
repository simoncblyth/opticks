#include "GLoader.hh"

#include "GVector.hh"
#include "GMergedMesh.hh"
#include "GBoundaryLib.hh"
#include "GBoundaryLibMetadata.hh"
#include "GColorizer.hh"
#include "GItemIndex.hh"
#include "GMaterial.hh"

#include "GGeo.hh"
#include "GCache.hh"
#include "GColors.hh"
#include "GColorMap.hh"

// npy-
#include "Lookup.hpp"
#include "Types.hpp"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void GLoader::load(bool verbose)
{
    GCache* m_cache = m_ggeo->getCache();  // prep for move into GGeo

    const char* idpath = m_cache->getIdPath() ;

    LOG(info) << "GLoader::load start " 
              << " idpath " << idpath 
              ;

    if(!m_ggeo->isLoaded()) 
    {
        LOG(info) << "GLoader::load slow loading using m_loader_imp (disguised AssimpGGeo) " ;

        m_ggeo->loadFromG4DAE(); 

        GBoundaryLib* blib = m_ggeo->getBoundaryLib();

        // material customization must be done prior to creating buffers, as they contain the customized indices ?
        m_materials = blib->getMaterials();  
        m_materials->loadIndex("$HOME/.opticks"); 

        // material/surface indices obtained from GItemIndex::getIndexLocal(shortname)
        blib->createWavelengthAndOpticalBuffers();

        m_metadata = blib->getMetadata();
        m_surfaces = blib->getSurfaces();   

        GColorMap* sixc = GColorMap::load("$HOME/.opticks", "GSurfaceIndexColors.json");
        m_surfaces->setColorMap(sixc);   
        m_surfaces->setColorSource(m_ggeo->getColors());

        m_ggeo->save(idpath );

        LOG(info) << "GLoader::load saving to cache directory " << idpath ;

        m_metadata->save(idpath);
        m_materials->save(idpath);
        m_surfaces->save(idpath);
    } 
    else
    {
        LOG(info) << "GLoader::load loading from cache directory " << idpath ;

        m_ggeo->loadFromCache() ; 

        GBoundaryLib* blib = m_ggeo->getBoundaryLib();

        m_metadata = blib->getMetadata() ; 

        m_materials  = GItemIndex::load(idpath, "GMaterialIndex"); 
        m_surfaces   = GItemIndex::load(idpath, "GSurfaceIndex");

        if(verbose)
            m_materials->dump("GLoader::load original materials from GItemIndex::load(idpath, \"GMaterialIndex\") ");
    } 


    // argh, Lookup  loads ChromaMaterialMap.json, GBoundaryLibMetadataMaterialMap.json 
    // ie more GBoundaryLib entanglement
    //
    m_lookup = new Lookup() ; 
    m_lookup->create(idpath);   
    m_lookup->dump("GLoader::load");  



    Index* idx = m_cache->getTypes()->getFlagsIndex() ;    
    m_flags = new GItemIndex( idx );   

    m_flags->setColorMap(GColorMap::load("$HOME/.opticks", "GFlagIndexColors.json"));    // itemname => colorname 
    m_materials->setColorMap(GColorMap::load("$HOME/.opticks", "GMaterialIndexColors.json")); 
    m_surfaces->setColorMap(GColorMap::load("$HOME/.opticks", "GSurfaceIndexColors.json"));   

    m_materials->setLabeller(GItemIndex::COLORKEY);
    m_surfaces->setLabeller(GItemIndex::COLORKEY);
    m_flags->setLabeller(GItemIndex::COLORKEY);
    
    GColors* colors = m_ggeo->getColors();
    m_materials->setColorSource(colors);
    m_surfaces->setColorSource(colors);
    m_flags->setColorSource(colors);

    m_surfaces->formTable();
    m_flags->formTable(); 
    m_materials->formTable();

    std::vector<unsigned int>& material_codes = m_materials->getCodes() ; 
    std::vector<unsigned int>& flag_codes     = m_flags->getCodes() ; 

    colors->setupCompositeColorBuffer( material_codes, flag_codes  );
    
}

