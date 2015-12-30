#include "App.hh"

int main(int argc, char** argv)
{
    App app("GGEOVIEW_", argc, argv);     

    // NumpyEvt created in App::config 
    app.configure(argc, argv);  
    if(app.isExit()) exit(EXIT_SUCCESS);

    // creates GGeo instance, loads, potentially modifies for (--test) and registers geometry
    app.loadGeometry();      
    if(app.isExit()) exit(EXIT_SUCCESS);


    bool nooptix = app.hasOpt("nooptix");
    bool noindex = app.hasOpt("noindex");
    bool noevent = app.hasOpt("noevent");
    bool save    = app.hasOpt("save");
    bool load    = app.hasOpt("load");

    if(load) save = false ; 

    if(!nooptix && !load)
    {
        app.loadGenstep();

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


    app.cleanup();

    exit(EXIT_SUCCESS);
}

