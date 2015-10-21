#include "App.hh"

int main(int argc, char** argv)
{
    App app("GGEOVIEW_", argc, argv); 
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.prepareScene();

    app.loadGeometry();
    if(app.isExit()) exit(EXIT_SUCCESS);

    app.configureGeometry();

    app.uploadGeometry();

    app.prepareOptiX();

    app.prepareGUI();

    app.renderLoop();

    app.cleanup();

    exit(EXIT_SUCCESS);
}

