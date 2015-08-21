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

#define OPTIX 1

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
#include "DynamicDefine.hh"


// TODO numpyserver-
#ifdef NPYSERVER
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"
#endif

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "NumpyEvt.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "Lookup.hpp"
// #include "Sensor.hpp"
#include "G4StepNPY.hpp"
#include "PhotonsNPY.hpp"
#include "RecordsNPY.hpp"
#include "BoundariesNPY.hpp"
#include "SequenceNPY.hpp"
#include "Types.hpp"
#include "Index.hpp"
#include "stringutil.hpp"

#include "Timer.hpp"
#include "Times.hpp"
#include "Parameters.hpp"
#include "Report.hpp"

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
#include "OEngine.hh"
#include "RayTraceConfig.hh"

// thrustrap-
#include "ThrustIdx.hh"
#include "ThrustHistogram.hh"
#include "ThrustArray.hh"

#define GLMVEC4(g) glm::vec4((g).x,(g).y,(g).z,(g).w) 


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
    logging_init();

    Parameters p ; 
    Timer t("main") ; 
    t.setVerbose(true);
    t.start();

    GCache cache("GGEOVIEW_") ; 
    const char* idpath = cache.getIdPath();
    bool juno          = cache.isJuno();
    const char* det    = cache.getDetector();


    const char* shader_dir = getenv("SHADER_DIR"); 
    const char* shader_incl_path = getenv("SHADER_INCL_PATH"); 

    Scene scene(shader_dir, shader_incl_path) ;

    Composition composition ;   
    Frame frame ;
    Bookmarks bookmarks ; 
    Interactor interactor ; 
#ifdef NPYSERVER
    numpydelegate delegate ; 
#endif
    NumpyEvt evt ;

    interactor.setFrame(&frame);
    interactor.setScene(&scene);

    interactor.setComposition(&composition);
    composition.setScene(&scene);
    bookmarks.setComposition(&composition);
    bookmarks.setScene(&scene);

    interactor.setBookmarks(&bookmarks);
    scene.setNumpyEvt(&evt);

    //t("wiring");  // minimal 0.001

    Cfg cfg("umbrella", false) ; // collect other Cfg objects
    FrameCfg<Frame>* fcfg = new FrameCfg<Frame>("frame", &frame,false);
    cfg.add(fcfg);
#ifdef NPYSERVER
    cfg.add(new numpydelegateCfg<numpydelegate>("numpydelegate", &delegate, false));
#endif

    cfg.add(new SceneCfg<Scene>(           "scene",       &scene,                      true));
    cfg.add(new RendererCfg<Renderer>(     "renderer",    scene.getGeometryRenderer(), true));
    cfg.add(new InteractorCfg<Interactor>( "interactor",  &interactor,                 true));
    composition.addConfig(&cfg); 

    cfg.commandline(argc, argv);
    t.setCommandLine(cfg.getCommandLine()); 

#ifdef NPYSERVER
    delegate.liveConnect(&cfg); // hookup live config via UDP messages
    delegate.setNumpyEvt(&evt); // allows delegate to update evt when NPY messages arrive
#endif

    if(fcfg->hasOpt("idpath")) std::cout << idpath << std::endl ;
    if(fcfg->hasOpt("help"))   std::cout << cfg.getDesc() << std::endl ;
    if(fcfg->isAbort()) exit(EXIT_SUCCESS); 

    bool fullscreen = fcfg->hasOpt("fullscreen");
    bool nooptix    = fcfg->hasOpt("nooptix");
    bool nogeocache = fcfg->hasOpt("nogeocache");
    bool noindex    = fcfg->hasOpt("noindex");
    bool noviz      = fcfg->hasOpt("noviz");
    bool nopropagate = fcfg->hasOpt("nopropagate");
    bool compute    = fcfg->hasOpt("compute");
    bool geocenter  = fcfg->hasOpt("geocenter");

    // x,y native 15inch retina resolution z: pixel factor (2: for retina)   x,y will be scaled down by the factor
    // pixelfactor 2 makes OptiX render at retina resolution
    // TODO: use GLFW to pluck the video mode screen size
    //
    // TODO: rationalize size setting, 
    //       currently gets set in frame, then overriden by value 
    //       from composition
    //       the below is a bandage workaround so that
    //       sizes from commandline are honoured 
    //
    //    ggv --size 640,480,2 
    //
    glm::uvec4 size ;
    if(fcfg->hasOpt("size")) size = frame.getSize() ;
    else if(fullscreen)      size = glm::uvec4(2880,1800,2,0) ;
    else                     size = glm::uvec4(2880,1704,2,0) ;  // 1800-44-44px native height of menubar  

    composition.setSize( size );

    frame.setInteractor(&interactor);      
    frame.setComposition(&composition);
    frame.setScene(&scene);
    frame.setTitle("GGeoView");
    frame.setFullscreen(fullscreen);

#ifdef NPYSERVER
    numpyserver<numpydelegate> server(&delegate); // connect to external messages 
#endif


    // dynamic define for use by GLSL shaders
    const char* shader_dynamic_dir = getenv("SHADER_DYNAMIC_DIR"); 
    DynamicDefine dd(shader_dynamic_dir, "dynamic.h");
    dd.add("MAXREC",fcfg->getRecordMax());    
    dd.add("MAXTIME",fcfg->getTimeMax());    
    dd.write();

    scene.init();  // reading shader source and creating renderers
    scene.setComposition(&composition);    

    frame.init();  // creates OpenGL context
    t("createOpenGLContext"); 
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
    loader.setInstanced(true); // find repeated geometry 
    loader.setRepeatIndex(fcfg->getRepeatIndex());
    loader.setTypes(&types);
    loader.setCache(&cache);
    loader.setImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr
    loader.load(nogeocache);

    p.add<int>("repeatIdx", loader.getRepeatIndex() );

    t("loadGeometry"); 

    GItemIndex* materials = loader.getMaterials();
    types.setMaterialsIndex(materials->getIndex());

    GBuffer* colorbuffer = loader.getColorBuffer();  // composite buffer 0+:materials,  32+:flags
    scene.uploadColorBuffer(colorbuffer);   // oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"

   
    GGeo* gg = loader.getGGeo();

    GMergedMesh* mm0 = NULL ; 
    unsigned int nmm = gg->getNumMergedMesh();
    for(unsigned int i=0 ; i < nmm ; i++)
    { 
        GMergedMesh* mm = gg->getMergedMesh(i); 
        if(i == 0)
        {
            mm0 = mm ; 
            unsigned int target = 0 ; 
            gfloat4 ce = mm->getCenterExtent(target);
            composition.setDomainCenterExtent(ce);     // index 0 corresponds to entire geometry

            LOG(info) << "main mm ce: " 
                      << " x " << ce.x
                      << " y " << ce.y
                      << " z " << ce.z
                      << " w " << ce.w
                      ;
        }
        scene.uploadGeometry(mm);  // TODO: support more than 1 instance geometry 
    
        if(i == 0)
        {
            scene.setTarget(0); // have to do in loop as currently uploadGeometry stomps on scene.m_geometry 
        }
    }

    bookmarks.load(idpath); 

    GBoundaryLib* blib = loader.getBoundaryLib();
 
    composition.setTimeDomain( gfloat4(0.f, fcfg->getTimeMax(), 0.f, 0.f) );  
    composition.setColorDomain( gfloat4(0.f, colorbuffer->getNumItems(), 0.f, 0.f));

    p.add<float>("timeMax",composition.getTimeDomain().y  ); 
    

    GBoundaryLibMetadata* meta = loader.getMetadata(); 
    std::map<int, std::string> boundaries = meta->getBoundaryNames();

    if(nogeocache){
        LOG(info) << "ggeoview/main.cc early exit due to --nogeocache/-G option " ; 
        exit(EXIT_SUCCESS); 
    }

    const char* typ ; 
    if(     fcfg->hasOpt("cerenkov"))      typ = "cerenkov" ;
    else if(fcfg->hasOpt("scintillation")) typ = "scintillation" ;
    else                                   typ = "cerenkov" ;

    std::string tag_ = fcfg->getEventTag();
    const char* tag = tag_.empty() ? "1" : tag_.c_str()  ; 


    t("interpGeometry"); 

    NPY<float>* npy = NPY<float>::load(typ, tag, det ) ;
    p.add<std::string>("genstepAsLoaded",   npy->getDigestString()  );

    t("loadGenstep"); 

    G4StepNPY genstep(npy);    
    genstep.setLookup(loader.getMaterialLookup()); 
   
    if(juno)
    {
        LOG(warning) << "main: kludge skip genstep.applyLookup for JUNO " ;
    }
    else
    {   
        genstep.applyLookup(0, 2);   // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 
    }
 
    p.add<std::string>("genstepAfterLookup",   npy->getDigestString()  );


    evt.setMaxRec(fcfg->getRecordMax());          // must set this before setGenStepData to have effect

    evt.setGenstepData(npy, nooptix); 

    t("hostEvtAllocation"); 


    glm::vec4 mmce = GLMVEC4(mm0->getCenterExtent(0)) ;
    glm::vec4 gsce = evt["genstep.vpos"]->getCenterExtent();
    glm::vec4 uuce = geocenter ? mmce : gsce ;
    print(mmce, "main mmce");
    print(gsce, "main gsce");
    print(uuce, "main uuce");
    bool autocam = true ; 
    composition.setCenterExtent( uuce , autocam );

    scene.setRecordStyle( fcfg->hasOpt("alt") ? Scene::ALTREC : Scene::REC );    


#ifdef INTEROP
    // signal Rdr to use GL_DYNAMIC_DRAW
    CUDAInterop<unsigned char>* c_psel = new CUDAInterop<unsigned char>(evt.getPhoselData());
    CUDAInterop<unsigned char>* c_rsel = new CUDAInterop<unsigned char>(evt.getRecselData());
#endif

    composition.update();
    //composition.dumpAxisData("main:dumpAxisData");
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


    // TODO:  
    //   * pull out the OptiX engine renderer to be external, and fit in with the scene ?
    //   * extract core OptiX processing into separate class
    //   * generation should not depend on renderers OpenGL buffers
    //     but for OpenGL interop its expedient for now


    Photons* photons(NULL);
#ifdef OPTIX
    
    OEngine::Mode_t mode = compute ? OEngine::COMPUTE : OEngine::INTEROP ; 

    OEngine engine("GGeoView", mode) ;       
    engine.setFilename(idpath);

    // transitioning back to gg, for multiple sets of instanced 
    engine.setGGeo(gg);   
    //engine.setMergedMesh(mm0);  

    engine.setBoundaryLib(blib);   
    engine.setNumpyEvt(&evt);
    engine.setComposition(&composition);                 
    engine.setEnabled(!nooptix);
    engine.setBounceMax(fcfg->getBounceMax());  // 0:prevents any propagation leaving generated photons
    engine.setRecordMax(evt.getMaxRec());       // 1:to minimize without breaking machinery 

    interactor.setTouchable(&engine);

    int rng_max = getenvint("CUDAWRAP_RNG_MAX",-1);
    assert(rng_max >= 1e6); 
    engine.setRngMax(rng_max);

    p.add<std::string>("Type", typ );
    p.add<std::string>("Tag", tag );
    p.add<std::string>("Detector", det );

    p.add<unsigned int>("NumGensteps", evt.getNumGensteps());
    p.add<unsigned int>("RngMax",     engine.getRngMax() );
    p.add<unsigned int>("NumPhotons", evt.getNumPhotons());
    p.add<unsigned int>("NumRecords", evt.getNumRecords());
    p.add<unsigned int>("BounceMax", engine.getBounceMax() );
    p.add<unsigned int>("RecordMax", engine.getRecordMax() );


    LOG(info)<< " ******************* main.OptiXEngine::init creating OptiX context, when enabled *********************** " ;
    engine.init();  
    t("initOptiX"); 
    LOG(info) << "main.OptiXEngine::init DONE "; 

    // persisting domain allows interpretation of packed photon record NPY arrays 
    // from standalone NumPy
    NPYBase* domain = engine.getDomain(); 
    //if(domain) domain->save("domain", "1", det);

    //NPYBase* idomain = engine.getIDomain(); 
    //if(idomain) idomain->save("idomain", "1", det );


    if(nopropagate)
    {
        LOG(info)<< " ******************* (main) OptiXEngine::generate INHIBITED by -P/--nopropagate  *********************** " ;
    }
    else
    {
        LOG(info)<< " ******************* (main) OptiXEngine::generate + propagate  *********************** " ;
        engine.generate();     
        t("generatePropagate"); 
    }

    LOG(info) << "main.OptiXEngine::generate DONE "; 



    // if(engine.isEnabled())
    // {
        NPY<float>* dpho = evt.getPhotonData();
        Rdr::download(dpho);

        NPY<short>* drec = evt.getRecordData();
        Rdr::download(drec);

        NPY<unsigned long long>* dhis = evt.getSequenceData();
        Rdr::download(dhis);

        t("evtDownload"); 

        p.add<std::string>("photonData",   dpho->getDigestString()  );
        p.add<std::string>("recordData",   drec->getDigestString()  );
        p.add<std::string>("sequenceData", dhis->getDigestString()  );

        t("checkDigests"); 

        dpho->setVerbose();
        dpho->save("ox%s", typ,  tag, det);
        drec->setVerbose();
        drec->save("rx%s", typ,  tag, det);
        dhis->setVerbose();
        dhis->save("ph%s", typ,  tag, det);

        t("evtSave"); 

        BoundariesNPY bnd(dpho); 
        bnd.setTypes(&types);
        bnd.setBoundaryNames(boundaries);
        bnd.indexBoundaries();

        PhotonsNPY pho(dpho);
        pho.setTypes(&types);

        RecordsNPY rec(drec, evt.getMaxRec());
        rec.setTypes(&types);
        rec.setDomains((NPY<float>*)domain);

        t("boundaryIndex"); 

        GItemIndex* seqhis = NULL ; 
        GItemIndex* seqmat = NULL ; 

        LOG(warning) << "main: hardcode noindex as not working" ;
        noindex = true ; 
        if(!noindex)
        {
            optix::Buffer& sequence_buffer = engine.getSequenceBuffer() ;
            unsigned int num_elements = OptiXUtil::getBufferSize1D( sequence_buffer );  assert(num_elements == 2*evt.getNumPhotons());
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
         
            LOG(info) << "main: ThrustIndex ctor " 
                      << " num_elements " << num_elements 
                      << " sequence_itemsize " << sequence_itemsize 
                      << " phosel_itemsize " << phosel_itemsize 
                      << " recsel_itemsize " << recsel_itemsize 
                      ; 
            ThrustArray<unsigned long long> pseq(d_seqn, num_elements       , sequence_itemsize );   // input flag/material sequences
            ThrustArray<unsigned char>      psel(d_psel, num_elements       , phosel_itemsize   );   // output photon selection
            ThrustArray<unsigned char>      rsel(d_rsel, num_elements*maxrec, recsel_itemsize   );   // output record selection

            ThrustIdx<unsigned long long, unsigned char> idx(&psel, &pseq);

            idx.makeHistogram(0, "FlagSequence");   
            idx.makeHistogram(1, "MaterialSequence");   

            psel.repeat_to( maxrec, rsel );
            cudaDeviceSynchronize();

            t("sequenceIndex"); 

#ifdef INTEROP
            // declare that CUDA finished with buffers 
            c_rsel->CUDA_to_GL();
            c_psel->CUDA_to_GL();
#else
            // non-interop workaround download the Thrust created buffers into NPY, then copy them back to GPU with uploadSelection
            psel.download( evt.getPhoselData() );  
            rsel.download( evt.getRecselData() ); 

            //evt.getPhoselData()->save("phosel_%s", typ, tag, det);
            //evt.getRecselData()->save("recsel_%s", typ, tag, det);

            scene.uploadSelection();                 // upload NPY into OpenGL buffer, duplicating recsel on GPU

            t("selectionDownloadUpload"); 
#endif

            seqhis = new GItemIndex(idx.getHistogramIndex(0)) ;  
            seqmat = new GItemIndex(idx.getHistogramIndex(1)) ;  
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
        }

        photons = new Photons(&pho, &bnd, seqhis, seqmat ) ; // GUI jacket 
        scene.setPhotons(photons);
    // }    // OptiX engine is enabled, ie not --nooptix/-O

#endif // WITH_OPTIX

    glm::ivec4& recsel = composition.getRecSelect();

#ifdef GUI_
    GUI gui ;
    gui.setScene(&scene);
    gui.setPhotons(photons);
    gui.setComposition(&composition);
    gui.setBookmarks(&bookmarks);
    gui.setInteractor(&interactor);   // status line
    gui.setLoader(&loader);           // access to Material / Surface indices
    
    gui.init(window);
    gui.setupHelpText( cfg.getDescString() );

    bool* show_gui_window = interactor.getGuiModeAddress();
#endif
 
    //t("GUI prep");  // minimal 0.001

    t.stop();

    p.dump();
    t.dump();

    Report r ; 
    r.add(p.getLines()); 
    r.add(t.getLines()); 

    Times* ts = t.getTimes();
    ts->save("$IDPATH/times", Times::name(typ, tag).c_str());

    char rdir[128];
    snprintf(rdir, 128, "$IDPATH/report/%s/%s", tag, typ ); 
    r.save(rdir, Report::name(typ, tag).c_str());  // with timestamp prefix


    if(noviz)
    {
        LOG(info) << "ggeoview/main.cc early exit due to --noviz/-V option " ; 
        exit(EXIT_SUCCESS); 
    }
    LOG(info) << "enter runloop "; 

    // TODO: use GItemIndex ? for stats to make it persistable
    gui.setupStats(t.getStats());
    gui.setupParams(p.getLines());

    //frame.toggleFullscreen(true); causing blankscreen then segv
    frame.hintVisible(true);
    frame.show();
    LOG(info) << "after frame.show() "; 

    unsigned int count ; 

    while (!glfwWindowShouldClose(window))
    {
        frame.listen(); 
#ifdef NPYSERVER
        server.poll_one();  
#endif
        frame.render();

#ifdef OPTIX
        if(interactor.getOptiXMode()>0)
        { 
             engine.trace();
             engine.render();
        }
        else
#endif
        {
            scene.render();
        }

        count = composition.tick();

#ifdef GUI_
        gui.newframe();
        if(*show_gui_window)
        {
            gui.show(show_gui_window);

            if(photons)
            {
                glm::ivec4 sel = photons->getBoundaries()->getSelection() ;
                composition.setSelection(sel); 
                composition.getPick().y = sel.x ;   //  1st boundary 

                recsel.x = seqhis ? seqhis->getSelected() : 0 ; 
                recsel.y = seqmat ? seqmat->getSelected() : 0 ; 

                composition.setFlags(types.getFlags()); 
            }
            // maybe imgui edit selection within the composition imgui, rather than shovelling ?
            // BUT: composition feeds into shader uniforms which could be reused by multiple classes ?
        }
        gui.render();
#endif

        glfwSwapBuffers(window);
    }
#ifdef OPTIX
    engine.cleanUp();
#endif
#ifdef NPYSERVER
    server.stop();
#endif
#ifdef GUI_
    gui.shutdown();
#endif
    frame.exit();
    exit(EXIT_SUCCESS);
}

