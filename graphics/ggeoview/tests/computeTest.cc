#include "App.hh"

int main(int argc, char** argv)
{
    App app("GGEOVIEW_", argc, argv);       // NumpyEvt created in App::config 
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.loadGeometry();       // creates GGeo instance, loads, potentially modifies for (--test) and registers geometry
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.loadGenstep();

    app.uploadEvt();               // allocates GPU buffers with OpenGL glBufferData

    app.seedPhotonsFromGensteps(); // distributes genstep indices into the photons VBO using CUDA

    //  app.initRecords();             // zero records VBO using CUDA

    app.prepareOptiX();

    app.preparePropagator();

    app.propagate();

    app.downloadEvt();

    app.makeReport();

    app.cleanup();

    exit(EXIT_SUCCESS);
}

