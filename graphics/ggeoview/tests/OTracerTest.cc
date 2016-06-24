#include <cstdio>
#include "NGLM.hpp"

#include "PLOG.hh"

#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "ASIRAP_LOG.hh"
#include "MESHRAP_LOG.hh"
#include "OKGEO_LOG.hh"
#include "OGLRAP_LOG.hh"
#include "GGV_LOG.hh"


#include "App.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    BRAP_LOG_ ;
    NPY_LOG_ ;
    OKCORE_LOG_ ;
    GGEO_LOG_ ;
    ASIRAP_LOG_ ;
    MESHRAP_LOG_ ;
    OKGEO_LOG_ ;
    OGLRAP_LOG_ ;
    GGV_LOG_ ;
 

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


    printf("%s exit\n", argv[0] );

    exit(EXIT_SUCCESS);
}

