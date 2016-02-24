#include "App.hh"

int main(int argc, char** argv)
{
    App app("OPTICKS_", argc, argv);     

    app.initViz();

    app.configure(argc, argv);    // NumpyEvt created in App::config, 
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.prepareViz();      // setup OpenGL shaders and creates OpenGL context (the window)

    app.loadGeometry();    // creates GGeo instance, loads, potentially modifies for (--test) and registers geometry
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.uploadGeometryViz();      // Scene::uploadGeometry, hands geometry to the Renderer instances for upload


    bool optixviz = app.hasOpt("optixviz");
    bool nooptix = app.hasOpt("nooptix");
    bool noindex = app.hasOpt("noindex");
    bool noevent = app.hasOpt("noevent");
    bool save    = app.hasOpt("save");
    bool load    = app.hasOpt("load");

    if(load) save = false ;    // "load" trumps "save" 

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
                app.saveEvt();

                app.indexEvtOld();  // indexing that depends on downloading to host 
            }
        }

    }
    else if(load)
    {

        if(optixviz)
        {
            app.prepareOptiX();            // places geometry into OptiX context with OGeo 

            app.prepareOptiXViz();         // creates ORenderer, OTracer
        } 

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

