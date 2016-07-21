#include <cstdio>
#include <iostream>

#include "NGLM.hpp"
#include "App.hh"

#include "PLOG.hh"

#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "ASIRAP_LOG.hh"
#include "MESHRAP_LOG.hh"
#include "OKGEO_LOG.hh"
#include "OGLRAP_LOG.hh"

#ifdef WITH_OPTIX
#include "CUDARAP_LOG.hh"
#include "THRAP_LOG.hh"
#include "OXRAP_LOG.hh"
#include "OKOP_LOG.hh"
#include "OKGL_LOG.hh"
#endif

#include "GGV_LOG.hh"

/**

GGeoViewTest
=============

Executable providing geometry and event propagation visualizations.


**/


int main(int argc, char** argv)
{
    //PLOG_(argc, argv);
    PLOG_COLOR(argc, argv);

    SYSRAP_LOG__ ;
    BRAP_LOG__ ;
    NPY_LOG__ ;
    OKCORE_LOG__ ;
    GGEO_LOG__ ;
    ASIRAP_LOG__ ;
    MESHRAP_LOG__ ;
    OKGEO_LOG__ ;
    OGLRAP_LOG__ ;

#ifdef WITH_OPTIX
    CUDARAP_LOG__ ;
    THRAP_LOG__ ;
    OXRAP_LOG__ ;
    OKOP_LOG__ ;
    OKGL_LOG__ ;
#endif

    GGV_LOG__ ;
 

    App app("OPTICKS_", argc, argv);     

    app.initViz();

    app.configure(argc, argv);    // NumpyEvt created in App::config, 
    if(app.isExit()) 
    {
        std::cerr << "app exit after configure" << std::endl ;        
        exit(EXIT_SUCCESS);
    }

    app.prepareViz();      // setup OpenGL shaders and creates OpenGL context (the window)

    app.loadGeometry();    // creates GGeo instance, loads, potentially modifies for (--test) and registers geometry
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.uploadGeometryViz();      // Scene::uploadGeometry, hands geometry to the Renderer instances for upload



    bool load = app.hasOpt("load");

    bool nooptix = app.hasOpt("nooptix");

    if(!nooptix && !load)
    {
        app.loadGenstep();             // hostside load genstep into NumpyEvt

        app.targetViz();               // point Camera at gensteps 

        app.uploadEvtViz();            // allocates GPU buffers with OpenGL glBufferData


#ifdef WITH_OPTIX
        app.prepareOptiX();            // places geometry into OptiX context with OGeo 

        app.prepareOptiXViz();         // creates ORenderer, OTracer

        if(!app.hasOpt("noevent"))
        {
            app.setupEventInEngine();
 
            app.preparePropagator();       // creates OptiX buffers and OBuf wrappers as members of OPropagator

            app.seedPhotonsFromGensteps(); // distributes genstep indices into the photons buffer

            if(app.hasOpt("onlyseed"))
            {
                 std::cerr << "onlyseed exit" << std::endl ;   
                 exit(EXIT_SUCCESS);
            }   


            app.initRecords();             // zero records buffer

            app.propagate();

            if(!app.hasOpt("noindex")) 
            {
                app.indexEvt();

                app.indexPresentationPrep();
            } 

            if(app.hasOpt("save") && !app.hasOpt("load"))
            {
                app.saveEvt();

                app.indexEvtOld();  // indexing that depends on downloading to host 
            }
        }
#endif

    }
    else if(load)
    {

#ifdef WITH_OPTIX
        if(app.hasOpt("optixviz"))
        {
            app.prepareOptiX();            // places geometry into OptiX context with OGeo 

            app.prepareOptiXViz();         // creates ORenderer, OTracer

            app.setupEventInEngine();   // for indexing 
        } 
#endif

        app.loadEvtFromFile();

        // huh maybe need to indexEvt if the indices are not loaded, 
        // eg when running with cfg4- no indices are persisted by the save as 
        // do that without assuming OptiX available

        app.indexEvt() ; // this skips if already indexed, and it handles loaded evt 

        app.indexPresentationPrep();

        app.uploadEvtViz();               // allocates GPU buffers with OpenGL glBufferData
    }

    app.prepareGUI();

    app.renderLoop();

    app.cleanup();

    exit(EXIT_SUCCESS);
}

