#include "GLoader.hh"

// look no hands : no dependency on AssimpWrap 

#include "GVector.hh"
#include "GMergedMesh.hh"
#include "GBoundaryLib.hh"
#include "GSensorList.hh"
#include "GBoundaryLibMetadata.hh"
#include "GTraverse.hh"
#include "GColorizer.hh"
#include "GTreeCheck.hh"
#include "GTreePresent.hh"
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


void GLoader::load()
{
    Timer t("GLoader::load") ; 
    t.setVerbose(true);
    t.start();

    assert(m_cache);

    const char* idpath = m_cache->getIdPath() ;

    const char* path = m_cache->getPath() ;
    const char* query = m_cache->getQuery() ;
    const char* ctrl = m_cache->getCtrl() ;

    const char* envprefix = m_cache->getEnvPrefix() ;

    LOG(info) << "GLoader::load start " 
              << " idpath " << idpath 
              << " repeatidx " << m_repeatidx 
              ;

    m_colors = GColors::load("$HOME/.opticks","GColors.json");  // colorname => hexcode 


    // more flexible to pass in nascent ggeo rather than have it created
    // in multiple other places 
    // eg this allows setting up mesh joiner imp to be done at creation within 
    // AssimpGGeo by GGeo

    if(m_ggeo->isLoaded()) 
    {
        LOG(info) << "GLoader::load loading from cache directory " << idpath ;

        m_ggeo->loadFromCache() ; 

        t("load ggeo/mergedmesh"); 

        // TODO: move below into GGeo::load 

        m_boundarylib = GBoundaryLib::load(idpath);
        t("load boundarylib"); 
        m_metadata = m_boundarylib->getMetadata() ; 

        m_materials  = GItemIndex::load(idpath, "GMaterialIndex"); 
        m_surfaces   = GItemIndex::load(idpath, "GSurfaceIndex");
        m_meshes     = GItemIndex::load(idpath, "MeshIndex");

        t("load indices"); 
        m_materials->dump("GLoader::load original materials from GItemIndex::load(idpath, \"GMaterialIndex\") ");
    } 
    else
    {
        LOG(info) << "GLoader::load slow loading using m_loader_imp (disguised AssimpGGeo) " << envprefix ;


        m_ggeo->loadFromG4DAE(); 

        t("create m_ggeo from G4DAE"); 

        //m_ggeo->add_sensitive_surfaces();

        t("add_sensitive_surfaces"); 


        m_boundarylib = m_ggeo->getBoundaryLib();

        // material customization must be done prior to creating buffers, as they contain the customized indices ?
        m_materials = m_boundarylib->getMaterials();  
        m_materials->loadIndex("$HOME/.opticks"); 

        m_boundarylib->createWavelengthAndOpticalBuffers();

        t("createWavelengthAndOpticalBuffers"); 


        // avoid requiring specific scintillator name by picking the first material 
        // with the requisite properties
        m_ggeo->findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB"); 
 
        GPropertyMap<float>* scint = dynamic_cast<GPropertyMap<float>*>(m_ggeo->getScintillatorMaterial(0));  

        m_boundarylib->createReemissionBuffer(scint);

        t("createReemissionBuffer"); 


        m_metadata = m_boundarylib->getMetadata();

        m_surfaces = m_boundarylib->getSurfaces();   

        GColorMap* sixc = GColorMap::load("$HOME/.opticks", "GSurfaceIndexColors.json");
        m_surfaces->setColorMap(sixc);   
        m_surfaces->setColorSource(m_colors);


        if(m_instanced)
        { 
            m_treeanalyse = new GTreeCheck(m_ggeo);  // TODO: rename to GTreeAnalyse
            m_treeanalyse->traverse();   // spin over tree counting up progenyDigests to find repeated geometry 
            m_treeanalyse->labelTree();  // recursive setRepeatIndex on the GNode tree for each of the repeated bits of geometry
            t("TreeCheck"); 

            GMergedMesh* mergedmesh = m_ggeo->makeMergedMesh(0, NULL);  // ridx:0 rbase:NULL 
            //mergedmesh->reportMeshUsage( m_ggeo, "GLoader::load reportMeshUsage (global)");

            unsigned int numRepeats = m_treeanalyse->getNumRepeats();
            for(unsigned int ridx=1 ; ridx <= numRepeats ; ridx++)  // 1-based index
            {
                GBuffer* rtransforms    = m_treeanalyse->makeTransformsBuffer(ridx);
                GNode*   rbase          = m_treeanalyse->getRepeatExample(ridx) ; 
                GMergedMesh* mergedmesh = m_ggeo->makeMergedMesh(ridx, rbase); 
                mergedmesh->dumpSolids("GLoader::load dumpSolids");
                mergedmesh->setTransformsBuffer(rtransforms);
                //mergedmesh->reportMeshUsage( m_ggeo, "GLoader::load reportMeshUsage (instanced)");
            }


            t("makeRepeatTransforms"); 

            GTreePresent tpr(m_ggeo, 0, 100, 1000);   // top,depth_max,sibling_max
            tpr.traverse();
            //tpr.dump("GLoader::load GTreePresent");
            tpr.write(idpath);
            
            t("treePresent"); 

        }
        else
        {
            LOG(warning) << "GLoader::load instancing is inhibited " ;
            m_ggeo->makeMergedMesh(0, NULL);  // ridx:0 rbase:NULL 
        }


        t("create MergedMesh"); 


        // GColorizer needs full tree,  so have to use pre-cache

        GMergedMesh* mesh0 = m_ggeo->getMergedMesh(0);

        gfloat3* vertex_colors = mesh0->getColors();

        GColorizer czr( vertex_colors, m_ggeo, GColorizer::PSYCHEDELIC_NODE ); 

        czr.setColors(m_colors);
        czr.setSurfaces(m_surfaces);
        czr.setRepeatIndex(mesh0->getIndex()); 
        czr.traverse();

        t("GColorizer"); 



        m_ggeo->save(idpath );

        LOG(info) << "GLoader::load saving to cache directory " << idpath ;

        // TODO: consolidate persistency management of below inside m_ggeo

        m_metadata->save(idpath);
        m_materials->save(idpath);
        m_surfaces->save(idpath);
        //m_meshes->save(idpath);

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

    m_flags->setColorMap(GColorMap::load("$HOME/.opticks", "GFlagIndexColors.json"));    

    // itemname => colorname 
    m_materials->setColorMap(GColorMap::load("$HOME/.opticks", "GMaterialIndexColors.json")); 
    m_surfaces->setColorMap(GColorMap::load("$HOME/.opticks", "GSurfaceIndexColors.json"));   


    m_materials->setLabeller(GItemIndex::COLORKEY);
    m_surfaces->setLabeller(GItemIndex::COLORKEY);
    m_flags->setLabeller(GItemIndex::COLORKEY);

    m_materials->setColorSource(m_colors);
    m_surfaces->setColorSource(m_colors);
    m_flags->setColorSource(m_colors);


    // formTable is needed to construct labels and codes when not pulling a buffer
    m_surfaces->formTable();
    m_flags->formTable(); 
    m_materials->formTable();


    unsigned int colormax = 256 ; 
    m_colors->initCompositeColorBuffer(colormax);

    std::vector<unsigned int>& material_codes = m_materials->getCodes() ; 
    std::vector<unsigned int>& flag_codes     = m_flags->getCodes() ; 
    std::vector<unsigned int>& psychedelic_codes = m_colors->getPsychedelicCodes();

    assert(material_codes.size() < 32 );
    assert(flag_codes.size() < 32 );
    assert(psychedelic_codes.size() < colormax-32 );

    unsigned int material_color_offset = 0 ; 
    unsigned int flag_color_offset     = 32 ; 
    unsigned int psychedelic_color_offset = 64 ; 

    m_colors->addColors(material_codes, material_color_offset ) ;
    m_colors->addColors(flag_codes    , flag_color_offset ) ;  
    m_colors->addColors(psychedelic_codes , psychedelic_color_offset ) ;  

    m_color_buffer = m_colors->getCompositeBuffer();

    m_color_domain.y = m_color_buffer->getNumItems() ;
    m_color_domain.z = float(psychedelic_codes.size()) ; 

    //m_colors->dumpCompositeBuffer("GLoader::load");

    LOG(info) << "GLoader::load done " << idpath ;
    //assert(m_mergedmesh);
    t.stop();
    t.dump();
}




