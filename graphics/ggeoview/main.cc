#include <stdlib.h>  //exit()
#include <stdio.h>

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"
#include "FrameCfg.hh"
#include "Scene.hh"
#include "SceneCfg.hh"
#include "Renderer.hh"
#include "RendererCfg.hh"
#include "Interactor.hh"
#include "InteractorCfg.hh"

#include "Bookmarks.hh"
#include "Composition.hh"
#include "Geometry.hh"
#include "Rdr.hh"
#include "Texture.hh"

// numpyserver-
#include "numpydelegate.hpp"
#include "numpydelegateCfg.hpp"
#include "numpyserver.hpp"

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "NumpyEvt.hpp"
#include "VecNPY.hpp"
#include "MultiVecNPY.hpp"
#include "Lookup.hpp"
#include "G4StepNPY.hpp"
#include "stringutil.hpp"


#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ggeo-
#include "GGeo.hh"
#include "GMergedMesh.hh"




#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


// optixrap-
#include "OptiXEngine.hh"
#include "RayTraceConfig.hh"


// cudawrap-
//using namespace optix ; 
//#include "cuRANDWrapper.hh"
//#include "curand.h"
//#include "curand_kernel.h"


void logging_init()
{
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
    );
}

int main(int argc, char** argv)
{
    logging_init();
    LOG(info) << argv[0] ; 

    Frame frame ;
    Composition composition ;   
    Bookmarks bookmarks ; 
    Interactor interactor ; 
    numpydelegate delegate ; 
    NumpyEvt evt ;
    Scene scene ;

    composition.setPixelFactor(2); // 2: makes OptiX render at retina resolution

    frame.setInteractor(&interactor);             // GLFW key/mouse events from frame to interactor and on to composition constituents

    interactor.setComposition(&composition);
    scene.setComposition(&composition);    
    bookmarks.setComposition(&composition);

    interactor.setBookmarks(&bookmarks);
    scene.setNumpyEvt(&evt);
    bookmarks.setScene(&scene);


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

    if(cfg["frame"]->hasOpt("help"))  std::cout << cfg.getDesc() << std::endl ;
    if(cfg["frame"]->isAbort()) exit(EXIT_SUCCESS); 


    numpyserver<numpydelegate> server(&delegate); // connect to external messages 

    frame.gl_init_window("GGeoView", composition.getWidth(),composition.getHeight());    // creates OpenGL context 

    bool nogeocache = cfg["frame"]->hasOpt("nogeocache");
    const char* idpath = scene.loadGeometry("GGEOVIEW_", nogeocache) ; 
    bookmarks.load(idpath); 
    GMergedMesh* mm = scene.getMergedMesh(); 


    // hmm would be better placed into a NumpyEvtCfg 
    const char* typ ; 
    if(     cfg["frame"]->hasOpt("cerenkov"))      typ = "cerenkov" ;
    else if(cfg["frame"]->hasOpt("scintillation")) typ = "scintillation" ;
    else                                           typ = "cerenkov" ;

    std::string tag_ = fcfg->getEventTag();
    const char* tag = tag_.empty() ? "1" : tag_.c_str()  ; 

    NPY* npy = NPY::load(typ, tag) ;

    G4StepNPY genstep(npy);    
    Lookup lookup ; 
    lookup.create(idpath);
    genstep.setLookup(&lookup); 
    genstep.applyLookup(0, 2); // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 

    evt.setGenstepData(npy); 

    scene.loadEvt();
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
    engine.setEnabled(interactor.getOptiXMode()>-1);

    int rng_max = getenvint("CUDAWRAP_RNG_MAX",-1);
    assert(rng_max >= 1e6); 
    engine.setRngMax(rng_max);
    engine.init();  // creates OptiX context, when enabled

    engine.generate();

    NPY* photons = evt.getPhotonData();
    Rdr::download(photons);
    char otyp[64];
    snprintf(otyp, 64, "ox%s", typ );
    photons->save(otyp, tag);


 
    GLFWwindow* window = frame.getWindow();
    LOG(info) << "enter runloop "; 
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

        glfwSwapBuffers(window);
    }
    engine.cleanUp();
    server.stop();
    frame.exit();
    exit(EXIT_SUCCESS);
}

