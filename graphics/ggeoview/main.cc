#include "App.hh"

int main(int argc, char** argv)
{
    App app("GGEOVIEW_", argc, argv);       // NumpyEvt created in App::config 
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.prepareScene();       // setup OpenGL shaders and creates OpenGL context (the window)

    app.loadGeometry();       // creates GGeo instance, loads, potentially modifies for (--test) and registers geometry
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.configureGeometry();   // setup geometry slicing for debug 

    app.uploadGeometry();      // Scene::uploadGeometry, hands geometry to the Renderer instances for upload

    bool nooptix = app.hasOpt("nooptix");
    bool noindex = app.hasOpt("noindex");
    bool noevent = app.hasOpt("noevent");
    bool save    = app.hasOpt("save");
    bool load    = app.hasOpt("load");

    if(load) save = false ; 

    if(!nooptix && !load)
    {
        app.loadGenstep();

        app.uploadEvt();               // allocates GPU buffers with OpenGL glBufferData

        app.seedPhotonsFromGensteps(); // distributes genstep indices into the photons VBO using CUDA

        app.initRecords();             // zero records VBO using CUDA

        app.prepareOptiX();

        if(!noevent)
        {
            app.preparePropagator();

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

        app.uploadEvt();               // allocates GPU buffers with OpenGL glBufferData
    }

    app.prepareGUI();

    app.renderLoop();

    app.cleanup();

    exit(EXIT_SUCCESS);
}

