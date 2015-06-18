#include <stdlib.h>  //exit()
#include <stdio.h>

#include "define.h"

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"
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
#include "stringutil.hpp"

// bregex-
#include "regexsearch.hh"


#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ggeo-
#include "GGeo.hh"
#include "GMergedMesh.hh"
#include "GBoundaryLibMetadata.hh"
#include "GLoader.hh"


#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>

#include <boost/log/trivial.hpp>
#include "boost/log/utility/setup.hpp"
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


// optixrap-
#include "OptiXEngine.hh"
#include "RayTraceConfig.hh"

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
    const char* prefix = "GGEOVIEW_" ;
    const char* idpath = GLoader::identityPath(prefix) ;
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

    numpyserver<numpydelegate> server(&delegate); // connect to external messages 


    frame.init();  // creates OpenGL context
    LOG(info) << "main: frame.init DONE "; 
    GLFWwindow* window = frame.getWindow();

    bool nooptix = fcfg->hasOpt("nooptix");
    bool nogeocache = fcfg->hasOpt("nogeocache");

    const char* idpath_ = scene.loadGeometry(prefix, nogeocache) ; 
    scene.setTarget(0);
    assert(strcmp(idpath_,idpath) == 0);  // TODO: use idpath in the loading, needs modif in AssimpGGeo::load
    bookmarks.load(idpath); 


    GMergedMesh* mm = scene.getMergedMesh(); 
    composition.setDomainCenterExtent(mm->getCenterExtent(0));  // index 0 corresponds to entire geometry
    composition.setTimeDomain( gfloat4(0.f, MAXTIME, 0.f, 0.f) );

    GBoundaryLibMetadata* meta = scene.getMetadata(); 
    std::map<int, std::string> boundaries = meta->getBoundaryNames();

    // hmm would be better placed into a NumpyEvtCfg 
    const char* typ ; 
    if(     fcfg->hasOpt("cerenkov"))      typ = "cerenkov" ;
    else if(fcfg->hasOpt("scintillation")) typ = "scintillation" ;
    else                                   typ = "cerenkov" ;

    std::string tag_ = fcfg->getEventTag();
    const char* tag = tag_.empty() ? "1" : tag_.c_str()  ; 



    NPY<float>* npy = NPY<float>::load(typ, tag) ;

    G4StepNPY genstep(npy);    

    Lookup lookup ; 
    lookup.create(idpath);

    // hmm probably this belongs with geometry loading 
    Sensor sensor ; 
    sensor.load(idpath, "idmap");


    genstep.setLookup(&lookup); 
    genstep.applyLookup(0, 2); // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 

    evt.setMaxRec( MAXREC );  // must set this before setGenStepData to have effect
    evt.setGenstepData(npy); 

    composition.setCenterExtent(evt["genstep.vpos"]->getCenterExtent());


    scene.setRecordStyle( fcfg->hasOpt("alt") ? Scene::ALTREC : Scene::REC );    
    scene.uploadEvt();
    LOG(info) << "main: scene.uploadEvt DONE "; 
    //
    // TODO:  
    //   * pull out the OptiX engine renderer to be external, and fit in with the scene ?
    //   * extract core OptiX processing into separate class
    //   * hmm generation should not depend on renderers OpenGL buffers
    //     but for OpenGL interop its expedient for now
    //
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
    engine.init();  // creates OptiX context, when enabled
    LOG(info) << "main: engine.init DONE "; 

    // persisting domain allows interpretation of packed photon record NPY arrays 
    // from standalone NumPy
    NPYBase* domain = engine.getDomain(); 
    if(domain) domain->save("domain", "1");

    engine.generate();
    LOG(info) << "main: engine.generate DONE "; 

    NPY<float>* photonData = evt.getPhotonData();
    Rdr::download(photonData);
    photonData->save("ox%s", typ,  tag);


    NPY<short>* rec = evt.getRecordData();
    Rdr::download(rec);
    rec->save("rx%s", typ,  tag);


    Photons photons(photonData) ;
    photons.setBoundaryNames(boundaries);
    photons.classify();
    photons.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    //photons.dumpFlags();

    scene.setPhotons(&photons);

#ifdef GUI_
    GUI gui ;
    gui.setScene(&scene);
    gui.setComposition(&composition);
    gui.setBookmarks(&bookmarks);
    gui.setInteractor(&interactor);   // status line

    gui.init(window);
    gui.setupHelpText( cfg.getDescString() );

    bool* show_gui_window = interactor.getGuiModeAddress();
#endif
 
    LOG(info) << "enter runloop "; 


    //frame.toggleFullscreen(true); causing blankscreen then segv
    frame.hintVisible(true);
    frame.show();

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

        composition.tick();

#ifdef GUI_
        gui.newframe();
        if(*show_gui_window)
        {
            gui.show(show_gui_window);

            glm::ivec4 sel = photons.getSelection() ;
            composition.setSelection(sel); 
            composition.getPick().y = sel.x ;   //  1st boundary 
            composition.setFlags(photons.getFlags()); 
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

