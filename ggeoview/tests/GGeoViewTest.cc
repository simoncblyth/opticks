
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
 

    App app(argc, argv);     

    app.configure();    // OpticksEvent created in App::config, 
    if(app.isExit()) exit(EXIT_SUCCESS);

    bool load = app.hasOpt("load");
    bool nooptix = app.hasOpt("nooptix");

    app.prepareViz();              // setup OpenGL shaders and creates OpenGL context (the window)

    app.loadGeometry();           // creates GGeo instance, loads, potentially modifies for (--test) and registers geometry
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.uploadGeometryViz();      // Scene::uploadGeometry, hands geometry to the Renderer instances for upload

#ifdef WITH_OPTIX
    if(!nooptix) app.prepareOptiX();    // places geometry into OptiX context with OGeo  and prepares OTracer, ORenderer
#endif


    if(!nooptix && !load)
    {
        app.loadGenstep();             // hostside load genstep into NumpyEvt

        app.targetViz();               // point Camera at gensteps 

        app.uploadEvtViz();            // allocates GPU buffers with OpenGL glBufferData

#ifdef WITH_OPTIX
        if(!app.hasOpt("noevent"))
        {
            //app.setupEventInEngine();
 
            app.preparePropagator();       // creates OptiX buffers and OBuf wrappers as members of OPropagator

            app.seedPhotonsFromGensteps(); // distributes genstep indices into the photons buffer

            if(app.hasOpt("onlyseed"))
            {
                 LOG(info) << "onlyseed exit" ;   
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
        //app.setupEventInEngine();      // for indexing, huh before loading 
#endif
        app.loadEvtFromFile();

        app.targetViz();                  // point Camera at gensteps 

        app.indexEvt() ;                 // this skips if already indexed, and it handles loaded evt 

        app.indexPresentationPrep();

        app.uploadEvtViz();               // allocates GPU buffers with OpenGL glBufferData
    }


    app.prepareGUI();

    app.renderLoop();

    app.cleanup();

    exit(EXIT_SUCCESS);
}

