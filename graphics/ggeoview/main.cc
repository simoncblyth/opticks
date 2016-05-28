#include "App.hh"
#include <cstdio>
#include <iostream>

int main(int argc, char** argv)
{
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



    bool nooptix = app.hasOpt("nooptix");
    bool save    = app.hasOpt("save");
    bool load    = app.hasOpt("load");
    if(load) save = false ;    // "load" trumps "save" 


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

            app.initRecords();             // zero records buffer

            app.propagate();

            if(!app.hasOpt("noindex")) 
            {
                app.indexEvt();

                app.indexPresentationPrep();
            } 

            if(save)
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

