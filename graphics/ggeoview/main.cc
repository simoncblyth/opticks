#include "App.hh"

int main(int argc, char** argv)
{
    App app("GGEOVIEW_", argc, argv);     

    app.initViz();

    // NumpyEvt created in App::config 
    app.configure(argc, argv);  
    if(app.isExit()) exit(EXIT_SUCCESS);

    // setup OpenGL shaders and creates OpenGL context (the window)
    app.prepareViz();      

    // creates GGeo instance, loads, potentially modifies for (--test) and registers geometry
    app.loadGeometry();      
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.uploadGeometryViz();      // Scene::uploadGeometry, hands geometry to the Renderer instances for upload


    bool nooptix = app.hasOpt("nooptix");
    bool noindex = app.hasOpt("noindex");
    bool noevent = app.hasOpt("noevent");
    bool save    = app.hasOpt("save");
    bool load    = app.hasOpt("load");

    if(load) save = false ; 

    if(!nooptix && !load)
    {
        app.loadGenstep();             // hostside load genstep into NumpyEvt

        app.targetViz();               // point Camera at gensteps 

        app.uploadEvtViz();            // allocates GPU buffers with OpenGL glBufferData

        app.prepareOptiX();            // places geometry into OptiX context with OGeo 

        app.prepareOptiXViz();         // creates ORenderer, OTracer


        if(!noevent)
        {
            app.preparePropagator();       // creates OptiX buffers and OBuf wrappers as members of OPropagator

            app.seedPhotonsFromGensteps(); // distributes genstep indices into the photons buffer

            app.initRecords();             // zero records buffer


            app.propagate();

            if(!noindex) 
            {
                app.indexEvt();

                app.indexPresentationPrep();
            } 

            if(save)
            {
                app.downloadEvt();

                app.indexEvtOld();  // indexing that depends on downloading to host 
            }
        }

        app.makeReport();
    }
    else if(load)
    {
        app.loadEvtFromFile();

        app.indexPresentationPrep();

        app.uploadEvtViz();               // allocates GPU buffers with OpenGL glBufferData
    }

    app.prepareGUI();

    app.renderLoop();

    app.cleanup();

    exit(EXIT_SUCCESS);
}

