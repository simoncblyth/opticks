#include <stdlib.h>  //exit()
#include <stdio.h>

#include "OptiXUtil.hh"
#include "define.h"

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"

// ggeoview-
//#define INTEROP 1
#ifdef INTEROP
#include "CUDAInterop.hh"
#endif

// oglrap-
#define GUI_ 1
#ifdef GUI_
#include "GUI.hh"
#endif

#include "FrameCfg.hh"
#include "Scene.hh"
#include "SceneCfg.hh"
#include "Renderer.hh"
#include "RendererCfg.hh"
#include "Interactor.hh"
#include "InteractorCfg.hh"

#include "Bookmarks.hh"
#include "Composition.hh"
#include "Rdr.hh"
#include "Texture.hh"
#include "Photons.hh"


// numpyserver-
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "NumpyEvt.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "Lookup.hpp"
#include "Sensor.hpp"
#include "G4StepNPY.hpp"
#include "PhotonsNPY.hpp"
#include "RecordsNPY.hpp"
#include "BoundariesNPY.hpp"
#include "SequenceNPY.hpp"
#include "Types.hpp"
#include "Index.hpp"
#include "stringutil.hpp"
#include "Timer.hpp"

// bregex-
#include "regexsearch.hh"

// glm-
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ggeo-
#include "GGeo.hh"
#include "GMergedMesh.hh"
#include "GBoundaryLibMetadata.hh"
#include "GLoader.hh"
#include "GCache.hh"
#include "GMaterialIndex.hh"

// assimpwrap
#include "AssimpGGeo.hh"


#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include "boost/log/utility/setup.hpp"
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

// optixrap-
#include "OptiXEngine.hh"
#include "RayTraceConfig.hh"

// thrustrap-
#include "ThrustIdx.hh"
#include "ThrustHistogram.hh"
#include "ThrustArray.hh"


void dump(float* f, const char* msg)
{
    if(!f) return ;

    printf("%s\n", msg);
    for(unsigned int i=0 ; i < 16 ; i++)
    {   
        if(i%4 == 0) printf("\n");
        printf(" %10.4f ", f[i] );
    }   
    printf("\n");
}


void logging_init()
{
   // see blogg-
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
    );

    boost::log::add_console_log(
        std::cerr, 
        boost::log::keywords::format = "[%TimeStamp%]: %Message%",
        boost::log::keywords::auto_flush = true
    );  

    boost::log::add_common_attributes();

}



int main(int argc, char** argv)
{
    Timer t ; 
    t.start();

    logging_init();
    GCache cache("GGEOVIEW_") ; 
    const char* idpath = cache.getIdPath();
    LOG(debug) << argv[0] ; 

    const char* shader_dir = getenv("SHADER_DIR"); 
    const char* shader_incl_path = getenv("SHADER_INCL_PATH"); 

    Scene scene(shader_dir, shader_incl_path) ;
    Composition composition ;   
    Frame frame ;
    Bookmarks bookmarks ; 
    Interactor interactor ; 
    numpydelegate delegate ; 
    NumpyEvt evt ;

    interactor.setFrame(&frame);
    interactor.setScene(&scene);

    interactor.setComposition(&composition);
    scene.setComposition(&composition);    
    composition.setScene(&scene);
    bookmarks.setComposition(&composition);
    bookmarks.setScene(&scene);

    interactor.setBookmarks(&bookmarks);
    scene.setNumpyEvt(&evt);

    t("wiring"); 

    Cfg cfg("umbrella", false) ; // collect other Cfg objects
    FrameCfg<Frame>* fcfg = new FrameCfg<Frame>("frame", &frame,false);
    cfg.add(fcfg);
    cfg.add(new numpydelegateCfg<numpydelegate>("numpydelegate", &delegate, false));

    cfg.add(new SceneCfg<Scene>(           "scene",       &scene,                      true));
    cfg.add(new RendererCfg<Renderer>(     "renderer",    scene.getGeometryRenderer(), true));
    cfg.add(new InteractorCfg<Interactor>( "interactor",  &interactor,                 true));
    composition.addConfig(&cfg); 

    cfg.commandline(argc, argv);
    delegate.liveConnect(&cfg); // hookup live config via UDP messages
    delegate.setNumpyEvt(&evt); // allows delegate to update evt when NPY messages arrive

    if(fcfg->hasOpt("idpath")) std::cout << idpath << std::endl ;
    if(fcfg->hasOpt("help"))   std::cout << cfg.getDesc() << std::endl ;
    if(fcfg->isAbort()) exit(EXIT_SUCCESS); 

    bool fullscreen = fcfg->hasOpt("fullscreen");
    bool nooptix = fcfg->hasOpt("nooptix");
    bool nogeocache = fcfg->hasOpt("nogeocache");
    bool noviz = fcfg->hasOpt("noviz");


    // x,y native 15inch retina resolution z: pixel factor (2: for retina)   x,y will be scaled down by the factor
    // pixelfactor 2 makes OptiX render at retina resolution
    // TODO: use GLFW to pluck the video mode screen size
    composition.setSize( fullscreen ? glm::uvec4(2880,1800,2,0) : glm::uvec4(2880,1704,2,0) );  // 1800-44-44px native height of menubar  
                                          //     1440  900

    // perhaps use an app class that just holds on to a instance of all objs ?
    frame.setInteractor(&interactor);             // GLFW key/mouse events from frame to interactor and on to composition constituents
    frame.setComposition(&composition);
    frame.setScene(&scene);
    frame.setTitle("GGeoView");
    frame.setFullscreen(fullscreen);

    t("configuration"); 
    numpyserver<numpydelegate> server(&delegate); // connect to external messages 
    t("numpyserver startup"); 

    frame.init();  // creates OpenGL context
    t("OpenGL context creation"); 
    LOG(info) << "main: frame.init DONE "; 

#ifdef INTEROP
    CUDAInterop<unsigned char>::init(); 
#endif

    GLFWwindow* window = frame.getWindow();

    Types types ;  
    types.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    Index* flags = types.getFlagsIndex(); 
    flags->setExt(".ini");
    //flags->save("/tmp");

    GLoader loader ;
    loader.setTypes(&types);
    loader.setCache(&cache);
    loader.setImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr
    loader.load(nogeocache);
    t("Geometry Loading"); 

    GItemIndex* materials = loader.getMaterials();
    types.setMaterialsIndex(materials->getIndex());

    GBuffer* colorbuffer = materials->getColorBuffer();  // TODO: combine colorbuffers for materials/surfaces/flags/... into one 
    scene.uploadColorBuffer(colorbuffer);
    scene.setGeometry(loader.getDrawable());
    scene.setTarget(0);

    bookmarks.load(idpath); 

    GMergedMesh* mm = loader.getMergedMesh(); 
    composition.setDomainCenterExtent(mm->getCenterExtent(0));  // index 0 corresponds to entire geometry
    composition.setTimeDomain( gfloat4(0.f, MAXTIME, 0.f, 0.f) );
    composition.setColorDomain( gfloat4(0.f, colorbuffer->getNumItems(), 0.f, 0.f));

    composition.dumpAxisData();


    GBoundaryLibMetadata* meta = loader.getMetadata(); 
    std::map<int, std::string> boundaries = meta->getBoundaryNames();


    if(nogeocache)
    {
        LOG(info) << "ggeoview/main.cc early exit due to --nogeocache/-G option " ; 
        exit(EXIT_SUCCESS); 
    }


    // hmm would be better placed into a NumpyEvtCfg 
    const char* typ ; 
    if(     fcfg->hasOpt("cerenkov"))      typ = "cerenkov" ;
    else if(fcfg->hasOpt("scintillation")) typ = "scintillation" ;
    else                                   typ = "cerenkov" ;

    std::string tag_ = fcfg->getEventTag();
    const char* tag = tag_.empty() ? "1" : tag_.c_str()  ; 

    t("Geometry Interp"); 


    NPY<float>* npy = NPY<float>::load(typ, tag) ;

    t("Genstep Loading"); 

    G4StepNPY genstep(npy);    
    genstep.setLookup(loader.getMaterialLookup()); 
    genstep.applyLookup(0, 2);   // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 

    evt.setMaxRec(MAXREC);          // must set this before setGenStepData to have effect
    evt.setGenstepData(npy); 

    t("Host Evt allocation"); 

    composition.setCenterExtent(evt["genstep.vpos"]->getCenterExtent());
    // is this domain used for photon record compression ?

    scene.setRecordStyle( fcfg->hasOpt("alt") ? Scene::ALTREC : Scene::REC );    



#ifdef INTEROP
    // signal Rdr to use GL_DYNAMIC_DRAW
    CUDAInterop<unsigned char>* c_psel = new CUDAInterop<unsigned char>(evt.getPhoselData());
    CUDAInterop<unsigned char>* c_rsel = new CUDAInterop<unsigned char>(evt.getRecselData());
#endif


    scene.uploadAxis();

    scene.uploadEvt();  // Scene, Rdr uploads orchestrated by NumpyEvt/MultiViewNPY

    t("uploadEvt"); 
    LOG(info) << "main: scene.uploadEvt DONE "; 

#ifdef INTEROP
    scene.uploadSelection();
    //c_psel->registerBuffer();
    //c_rsel->registerBuffer();
#else
    // non-interop workaround: defer uploadSelection until after indexing 
#endif


    //
    // TODO:  
    //   * pull out the OptiX engine renderer to be external, and fit in with the scene ?
    //   * extract core OptiX processing into separate class
    //   * hmm generation should not depend on renderers OpenGL buffers
    //     but for OpenGL interop its expedient for now


    //  creates OptiX buffers from the OpenGL buffer_id's 
    OptiXEngine engine("GGeoView") ;       
    engine.setFilename(idpath);
    engine.setMergedMesh(mm);   
    engine.setNumpyEvt(&evt);
    engine.setComposition(&composition);                 
    engine.setEnabled(!nooptix);
    engine.setBounceMax(fcfg->getBounceMax());  // 0:prevents any propagation leaving generated photons

    interactor.setTouchable(&engine);

    int rng_max = getenvint("CUDAWRAP_RNG_MAX",-1);
    assert(rng_max >= 1e6); 
    engine.setRngMax(rng_max);

    LOG(info)<< " ******************* main.OptiXEngine::init creating OptiX context, when enabled *********************** " ;
    engine.init();  
    t("OptiXEngine init"); 
    LOG(info) << "main.OptiXEngine::init DONE "; 

    // persisting domain allows interpretation of packed photon record NPY arrays 
    // from standalone NumPy
    NPYBase* domain = engine.getDomain(); 
    //if(domain) domain->save("domain", "1");

    //NPYBase* idomain = engine.getIDomain(); 
    //if(idomain) idomain->save("idomain", "1");

   
    LOG(info)<< " ******************* (main) OptiXEngine::generate + propagate  *********************** " ;
    engine.generate();     
    t("OptiXEngine generate, propagate"); 

    LOG(info) << "main.OptiXEngine::generate DONE "; 


    NPY<float>* dpho = evt.getPhotonData();
    Rdr::download(dpho);

    NPY<short>* drec = evt.getRecordData();
    Rdr::download(drec);

    NPY<NumpyEvt::Sequence_t>* dhis = evt.getSequenceData();
    Rdr::download(dhis);

    t("photon, record, sequence downloads"); 

    dpho->setVerbose();
    dpho->save("ox%s", typ,  tag);
    drec->setVerbose();
    drec->save("rx%s", typ,  tag );
    dhis->setVerbose();
    dhis->save("ph%s", typ,  tag );

    t("photon, record, sequence save"); 


    BoundariesNPY bnd(dpho); 
    bnd.setTypes(&types);
    bnd.setBoundaryNames(boundaries);
    bnd.indexBoundaries();

    PhotonsNPY pho(dpho);
    pho.setTypes(&types);

    RecordsNPY rec(drec, evt.getMaxRec());
    rec.setTypes(&types);
    rec.setDomains((NPY<float>*)domain);

    t("boundary indexing"); 


    optix::Buffer& sequence_buffer = engine.getSequenceBuffer() ;
    unsigned int num_elements = OptiXUtil::getBufferSize1D( sequence_buffer );  assert(num_elements == evt.getNumPhotons());
    unsigned int device_number = 0 ;  // maybe problem with multi-GPU
    unsigned long long* d_seqn = OptiXUtil::getDevicePtr<unsigned long long>( sequence_buffer, device_number ); 


#ifdef INTEROP
    // attempt to give CUDA access to mapped OpenGL buffer
    //unsigned char*      d_psel = c_psel->GL_to_CUDA();
    //unsigned char*      d_rsel = c_rsel->GL_to_CUDA();
    // attempt to give CUDA access to OptiX buffers which in turn are connected to OpenGL buffers
    unsigned char* d_psel = OptiXUtil::getDevicePtr<unsigned char>( engine.getPhoselBuffer(), device_number ); 
    unsigned char* d_rsel = OptiXUtil::getDevicePtr<unsigned char>( engine.getRecselBuffer(), device_number ); 
#else
    LOG(info)<< "main: non interop allocating new device buffers with ThrustArray " ;
    unsigned char*      d_psel = NULL ;    
    unsigned char*      d_rsel = NULL ;    
#endif

    unsigned int sequence_itemsize = evt.getSequenceData()->getShape(2) ; assert( 2 == sequence_itemsize );
    unsigned int phosel_itemsize   = evt.getPhoselData()->getShape(2)   ; assert( 4 == phosel_itemsize );
    unsigned int recsel_itemsize   = evt.getRecselData()->getShape(2)   ; assert( 4 == recsel_itemsize );
    unsigned int maxrec = evt.getMaxRec();
 
    LOG(info) << "main: ThrustIndex ctor " ; 
    ThrustArray<unsigned long long> pseq(d_seqn, num_elements       , sequence_itemsize );   // input flag/material sequences
    ThrustArray<unsigned char>      psel(d_psel, num_elements       , phosel_itemsize   );   // output photon selection
    ThrustArray<unsigned char>      rsel(d_rsel, num_elements*maxrec, recsel_itemsize   );   // output record selection

    ThrustIdx<unsigned long long, unsigned char> idx(&psel, &pseq);

    idx.makeHistogram(0, "FlagSequence");   
    idx.makeHistogram(1, "MaterialSequence");   

    psel.repeat_to( maxrec, rsel );
    cudaDeviceSynchronize();

    t("sequence indexing"); 

#ifdef INTEROP
    // declare that CUDA finished with buffers 
    c_rsel->CUDA_to_GL();
    c_psel->CUDA_to_GL();
#else
    // non-interop workaround download the Thrust created buffers into NPY, then copy them back to GPU with uploadSelection
    psel.download( evt.getPhoselData() );  
    rsel.download( evt.getRecselData() ); 

    //evt.getPhoselData()->save("phosel_%s", typ, tag);
    //evt.getRecselData()->save("recsel_%s", typ, tag);

    scene.uploadSelection();                 // upload NPY into OpenGL buffer, duplicating recsel on GPU

    t("selection download/upload"); 
#endif

    GItemIndex* seqhis = new GItemIndex(idx.getHistogramIndex(0)) ;  
    GItemIndex* seqmat = new GItemIndex(idx.getHistogramIndex(1)) ;  
    //seqhis->save(idpath);
    //seqmat->save(idpath);

    seqhis->setTitle("Photon Flag Sequence Selection");
    seqhis->setTypes(&types);
    seqhis->setLabeller(GItemIndex::HISTORYSEQ);
    seqhis->formTable();

    seqmat->setTitle("Photon Material Sequence Selection");
    seqmat->setTypes(&types);
    seqmat->setLabeller(GItemIndex::MATERIALSEQ);
    seqmat->formTable();

    if(noviz)
    {
        LOG(info) << "ggeoview/main.cc early exit due to --noviz/-V option " ; 
        exit(EXIT_SUCCESS); 
    }


    glm::ivec4& recsel = composition.getRecSelect();

    Photons photons(&pho, &bnd, seqhis, seqmat ) ; // GUI jacket 

    scene.setPhotons(&photons);  // seems to do little, maybe just for GUI

#ifdef GUI_
    GUI gui ;
    gui.setScene(&scene);
    gui.setComposition(&composition);
    gui.setBookmarks(&bookmarks);
    gui.setInteractor(&interactor);   // status line
    gui.setLoader(&loader);           // access to Material / Surface indices
    
    gui.init(window);
    gui.setupHelpText( cfg.getDescString() );

    bool* show_gui_window = interactor.getGuiModeAddress();
#endif
 
    t("GUI prep"); 
    LOG(info) << "enter runloop "; 

    t.stop();
    t.dump();
    // TODO: use GItemIndex ? for stats to make it persistable
    gui.setupStats(t.getStats());

    //frame.toggleFullscreen(true); causing blankscreen then segv
    frame.hintVisible(true);
    frame.show();

    unsigned int count ; 

    while (!glfwWindowShouldClose(window))
    {
        frame.listen(); 
        server.poll_one();  
        frame.render();

        if(interactor.getOptiXMode()>0)
        { 
             engine.trace();
             engine.render();
        }
        else
        {
            scene.render();
        }

        count = composition.tick();

#ifdef GUI_
        gui.newframe();
        if(*show_gui_window)
        {
            gui.show(show_gui_window);

            glm::ivec4 sel = bnd.getSelection() ;
            composition.setSelection(sel); 
            composition.getPick().y = sel.x ;   //  1st boundary 


            recsel.x = seqhis->getSelected(); 
            recsel.y = seqmat->getSelected(); 

            //if(count % 100 == 0) print(recsel, "main::recsel");

            composition.setFlags(types.getFlags()); 
            // maybe imgui edit selection within the composition imgui, rather than shovelling ?
            // BUT: composition feeds into shader uniforms which could be reused by multiple classes ?
        }
        gui.render();
#endif

        glfwSwapBuffers(window);
    }
    engine.cleanUp();
    server.stop();
#ifdef GUI_
    gui.shutdown();
#endif
    frame.exit();
    exit(EXIT_SUCCESS);
}

