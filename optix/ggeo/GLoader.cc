#include "GLoader.hh"

// look no hands : no dependency on AssimpWrap 

#include "GMergedMesh.hh"
#include "GBoundaryLib.hh"
#include "GSensorList.hh"
#include "GBoundaryLibMetadata.hh"
#include "GTraverse.hh"
#include "GColorizer.hh"
#include "GTreeCheck.hh"
#include "GItemIndex.hh"
#include "GBuffer.hh"
#include "GMaterial.hh"

#include "GGeo.hh"
#include "GCache.hh"
#include "GColors.hh"
#include "GColorMap.hh"

// npy-
#include "stringutil.hpp"
#include "Lookup.hpp"
#include "Types.hpp"
#include "Timer.hpp"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void GLoader::load(bool nogeocache)
{
    Timer t("GLoader::load") ; 
    t.setVerbose(true);
    t.start();

    assert(m_cache);
    const char* idpath = m_cache->getIdPath() ;
    const char* envprefix = m_cache->getEnvPrefix() ;

    LOG(info) << "GLoader::load start " 
              << " idpath " << idpath 
              << " nogeocache " << nogeocache  
              << " repeatidx " << m_repeatidx 
              ;

    fs::path geocache(idpath);

    if(fs::exists(geocache) && fs::is_directory(geocache) && !nogeocache ) 
    {
        LOG(info) << "GLoader::load loading from cache directory " << idpath ;

        m_ggeo = GGeo::load(idpath) ; 

        t("load ggeo/mergedmesh"); 

        // TODO: move below into GGeo::load 

        m_boundarylib = GBoundaryLib::load(idpath);
        t("load boundarylib"); 
        m_metadata = m_boundarylib->getMetadata() ; 

        m_materials  = GItemIndex::load(idpath, "GMaterialIndex"); // TODO: find common place for such strings, maybe Types.hpp
        m_surfaces   = GItemIndex::load(idpath, "GSurfaceIndex");
        m_meshes     = GItemIndex::load(idpath, "MeshIndex");

        t("load indices"); 
    } 
    else
    {
        LOG(info) << "GLoader::load slow loading using m_imp (disguised AssimpGGeo) " << envprefix ;

        m_ggeo = (*m_imp)(envprefix);      


        t("create m_ggeo from G4DAE"); 
        m_meshes = m_ggeo->getMeshIndex();  

        m_boundarylib = m_ggeo->getBoundaryLib();
        m_boundarylib->createWavelengthAndOpticalBuffers();

        t("createWavelengthAndOpticalBuffers"); 

        // moved Wavelength and Optical buffers to GBoundaryLib (from GMergedMesh)

        // avoid requiring specific scintillator name by picking the first material 
        // with the requisite properties
        m_ggeo->findScintillators("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB"); 
        //m_ggeo->dumpScintillators();
        GPropertyMap<float>* scint = dynamic_cast<GPropertyMap<float>*>(m_ggeo->getScintillator(0));  

        m_boundarylib->createReemissionBuffer(scint);

        t("createReemissionBuffer"); 

        GColors* source = GColors::load("$HOME/.opticks","GColors.json");  // colorname => hexcode 

        m_metadata = m_boundarylib->getMetadata();
        m_materials = m_boundarylib->getMaterials();  

        m_surfaces = m_boundarylib->getSurfaces();   
        m_surfaces->setColorMap(GColorMap::load(idpath, "GSurfaceIndexColors.json"));   
        m_surfaces->setColorSource(source);

        m_materials->loadIndex("$HOME/.opticks"); // customize GMaterialIndex

        m_ggeo->sensitize(idpath, "idmap");       // loads idmap and traverses nodes doing GSolid::setSensor for sensitve nodes

        t("sensitize"); 

        if(m_instanced)
        { 
            m_treeanalyse = new GTreeCheck(m_ggeo);  // TODO: rename to GTreeAnalyse
            m_treeanalyse->traverse();   // spin over tree counting up progenyDigests to find repeated geometry 
            m_treeanalyse->labelTree();  // recursive setRepeatIndex on the GNode tree for each of the repeated bits of geometry
            t("TreeCheck"); 

            GMergedMesh* mergedmesh = m_ggeo->makeMergedMesh(0, NULL);  // ridx:0 rbase:NULL 
            mergedmesh->reportMeshUsage( m_ggeo, "GLoader::load reportMeshUsage (global)");

            unsigned int numRepeats = m_treeanalyse->getNumRepeats();
            for(unsigned int ridx=1 ; ridx <= numRepeats ; ridx++)  // 1-based index
            {
                GBuffer* rtransforms    = m_treeanalyse->makeTransformsBuffer(ridx);
                GNode*   rbase          = m_treeanalyse->getRepeatExample(ridx) ; 
                GMergedMesh* mergedmesh = m_ggeo->makeMergedMesh(ridx, rbase); 
                mergedmesh->dumpSolids("GLoader::load dumpSolids");
                mergedmesh->setTransformsBuffer(rtransforms);
                mergedmesh->reportMeshUsage( m_ggeo, "GLoader::load reportMeshUsage (instanced)");
            }

            m_treeanalyse->dumpTree("GLoader::load dumpTree");
            t("makeRepeatTransforms"); 
        }
        else
        {
            assert(0);
            m_ggeo->makeMergedMesh(0, NULL);  // ridx:0 rbase:NULL 
        }

        m_ggeo->saveMergedMeshes(idpath );


        // if requested index of merged mesh exists already return it 
        // otherwise create using GMergedMesh::create(index, GGeo*) and return, 
        //
        //   ridx:0 is catchall global geometry
        //   ridx:1 is the first repeated geometry index, eg the PMTs
        //
        //
        //  ggv.sh  -G --repeatidx 1                        // create ridx 1 geocache, with single PMT subtree 
        //  ggv.sh  --noindex --repeatidx 1 --geocenter     // view the single PMT, need --geocenter as unplaced, nowhere near the genstep 
        //
  

        t("create MergedMesh"); 

        /*
        TODO: adapt GColorizer to new multi-mergedmesh regime 

        //m_mergedmesh->setColor(0.5,0.5,1.0); // this would scrub node colors

        gfloat3* target = m_mergedmesh->getColors();
        GColorizer czr( target, m_ggeo ); 
        czr.setSurfaces(m_surfaces);
        czr.setRepeatIndex(m_mergedmesh->getIndex()); 
        czr.traverse();

        t("GColorizer"); 

        */

        LOG(info) << "GLoader::load saving to cache directory " << idpath ;

        //m_mergedmesh->save(idpath); 

        m_metadata->save(idpath);
        m_materials->save(idpath);
        m_surfaces->save(idpath);
        m_meshes->save(idpath);

        m_boundarylib->saveIndex(idpath); 

        m_boundarylib->save(idpath);

        t("save geocache"); 
    } 
  
    // hmm not routing via cache 
    m_lookup = new Lookup() ; 
    m_lookup->create(idpath);
    //m_lookup->dump("GLoader::load");  

    Index* idx = m_types->getFlagsIndex() ;    
    m_flags = new GItemIndex( idx );     //GFlagIndex::load(idpath); 

    m_flags->setColorMap(GColorMap::load(idpath, "GFlagIndexColors.json"));    

    // itemname => colorname 
    m_materials->setColorMap(GColorMap::load(idpath, "GMaterialIndexColors.json")); 
    m_surfaces->setColorMap(GColorMap::load(idpath, "GSurfaceIndexColors.json"));   


    m_materials->setLabeller(GItemIndex::COLORKEY);
    m_surfaces->setLabeller(GItemIndex::COLORKEY);
    m_flags->setLabeller(GItemIndex::COLORKEY);

    m_colors = GColors::load("$HOME/.opticks","GColors.json"); // colorname => hexcode   TODO: remove this, it duplicates above source
    m_materials->setColorSource(m_colors);
    m_surfaces->setColorSource(m_colors);
    m_flags->setColorSource(m_colors);


    // formTable is needed to construct labels and codes when not pulling a buffer
    m_surfaces->formTable();
    m_flags->formTable(); 
    m_materials->formTable();


    m_colors->initCompositeColorBuffer(64);

    std::vector<unsigned int>& material_codes = m_materials->getCodes() ; 
    std::vector<unsigned int>& flag_codes     = m_flags->getCodes() ; 

    assert(material_codes.size() < 32 );
    assert(flag_codes.size() < 32 );

    unsigned int material_color_offset = 0 ; 
    unsigned int flag_color_offset     = 32 ; 

    m_colors->addColors(material_codes, material_color_offset ) ;
    m_colors->addColors(flag_codes    , flag_color_offset ) ;  

    m_color_buffer = m_colors->getCompositeBuffer();
    //m_colors->dumpCompositeBuffer("GLoader::load");


    LOG(info) << "GLoader::load done " << idpath ;
    //assert(m_mergedmesh);
    t.stop();
    t.dump();
}

void GLoader::Summary(const char* msg)
{
    printf("%s\n", msg);
    //m_mergedmesh->Summary("GLoader::Summary");
    //m_mergedmesh->Dump("GLoader::Summary Dump",10);
}



