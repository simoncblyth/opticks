#include <stdlib.h>  //exit()
#include <stdio.h>


// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h


#include <GL/glew.h>

#include "Frame.hh"

#define GUI 1
#ifdef GUI
#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"
//#include <GL/gl3w.h>
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
#include "boost/log/utility/setup.hpp"
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


// optixrap-
#include "OptiXEngine.hh"
#include "RayTraceConfig.hh"




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
    const char* idpath = Geometry::identityPath(prefix) ;
    LOG(info) << argv[0] ; 

    Frame frame ;
    Composition composition ;   
    Bookmarks bookmarks ; 
    Interactor interactor ; 
    numpydelegate delegate ; 
    NumpyEvt evt ;
    Scene scene ;

    composition.setPixelFactor(2); // 2: makes OptiX render at retina resolution
    frame.setPixelFactor(2);       // 2: makes OptiX render at retina resolution
    // NB another Coord2Pixel in Frame, TODO: unify all these


    // hmm needs some untangling... need to review purpose of each and do some method swapping ?
    frame.setInteractor(&interactor);             // GLFW key/mouse events from frame to interactor and on to composition constituents
    frame.setComposition(&composition);
    frame.setScene(&scene);
    interactor.setFrame(&frame);

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

    if(cfg["frame"]->hasOpt("idpath")) std::cout << idpath << std::endl ;
    if(cfg["frame"]->hasOpt("help"))   std::cout << cfg.getDesc() << std::endl ;
    if(cfg["frame"]->isAbort()) exit(EXIT_SUCCESS); 


    numpyserver<numpydelegate> server(&delegate); // connect to external messages 

    frame.gl_init_window("GGeoView", composition.getWidth(),composition.getHeight());    // creates OpenGL context 
    GLFWwindow* window = frame.getWindow();

#ifdef GUI
    //gl3wInit();
    ImGui_ImplGlfwGL3_Init(window, true);
    bool show_test_window = true;
    bool show_another_window = false;
    //ImVec4 clear_color = ImColor(114, 144, 154);
#endif


    bool nooptix = cfg["frame"]->hasOpt("nooptix");
    bool nogeocache = cfg["frame"]->hasOpt("nogeocache");
    const char* idpath_ = scene.loadGeometry(prefix, nogeocache) ; 
    assert(strcmp(idpath_,idpath) == 0);  // TODO: use idpath in the loading 
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
    engine.setEnabled(!nooptix);

    interactor.setTouchable(&engine);

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


 
    LOG(info) << "enter runloop "; 
    while (!glfwWindowShouldClose(window))
    {

#ifdef GUI
        ImGuiIO& io = ImGui::GetIO();
#endif
        frame.listen(); 
#ifdef GUI
        ImGui_ImplGlfwGL3_NewFrame();

        // 1. Show a simple window
        // Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appears in a window automatically called "Debug"
        {
            static float f = 0.0f;
            ImGui::Text("Hello, world!");
            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
            //ImGui::ColorEdit3("clear color", (float*)&clear_color);
            if (ImGui::Button("Test Window")) show_test_window ^= 1;
            if (ImGui::Button("Another Window")) show_another_window ^= 1;
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        }
#endif

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

#ifdef GUI
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        //glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        //glClear(GL_COLOR_BUFFER_BIT);
        ImGui::Render();
#endif

        glfwSwapBuffers(window);
    }
    engine.cleanUp();
    server.stop();

#ifdef GUI
    ImGui_ImplGlfwGL3_Shutdown();
#endif
    frame.exit();
    exit(EXIT_SUCCESS);
}

