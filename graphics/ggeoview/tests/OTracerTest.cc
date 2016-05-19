#include "App.hh"

int main(int argc, char** argv)
{
    App app("OPTICKS_", argc, argv); 

    app.initViz();

    app.configure(argc, argv);
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.prepareViz();

    app.loadGeometry();
    if(app.isExit()) exit(EXIT_SUCCESS);


    app.uploadGeometryViz();


#ifdef WITH_OPTIX
    app.prepareOptiX();

    app.prepareOptiXViz();
#endif


    app.prepareGUI();

    app.renderLoop();

    app.cleanup();

    exit(EXIT_SUCCESS);
}

