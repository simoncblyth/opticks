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

    bool nooptix = app.hasOpt("nooptix");
    bool noindex = app.hasOpt("noindex");
    bool noevent = app.hasOpt("noevent");
    bool save    = app.hasOpt("save");

    if(!nooptix)
    {

        app.loadGenstep();

        app.uploadEvt();    // allocates GPU buffers with OpenGL glBufferData

        app.seedPhotonsFromGensteps();

        app.initRecords();

        app.prepareOptiX();

        if(!noevent)
        {
            app.preparePropagator();

            app.propagate();

            if(!noindex) app.indexEvt();

            if(save)
            {

                app.downloadEvt();

                app.indexEvtOld();  // indexing that depends on downloading to host 
            }
        }

        app.makeReport();
    }

    app.prepareGUI();

    app.renderLoop();

    app.cleanup();

    exit(EXIT_SUCCESS);
}

