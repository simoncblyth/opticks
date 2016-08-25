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

    BRAP_LOG__ ;
    NPY_LOG__ ;
    OKCORE_LOG__ ;
    GGEO_LOG__ ;
    ASIRAP_LOG__ ;
    MESHRAP_LOG__ ;
    OKGEO_LOG__ ;
    OGLRAP_LOG__ ;
    GGV_LOG__ ;
 

    App app(argc, argv); 


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

