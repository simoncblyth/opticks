
#include "Opticks.hh"
#include "OpticksHub.hh"

#include "GGeoTest.hh"

#include "PLOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "OKGEO_LOG.hh"
#include "GGEO_LOG.hh"


int main(int argc, char** argv)
{
    //PLOG_(argc, argv);
    PLOG_COLOR(argc, argv);

    NPY_LOG__ ;
    OKCORE_LOG__ ;
    OKGEO_LOG__ ;
    GGEO_LOG__ ;

    //const char* funcname = "tboolean-torus--" ;
    const char* funcname = "tboolean-media--" ;
    //const char* funcname = "tboolean-nonexisting--" ;

    Opticks ok(argc, argv, GGeoTest::MakeArgForce(funcname, "--dbgsurf --dbgbnd") );

    OpticksHub hub(&ok);      

    if(hub.getErr()) LOG(fatal) << "hub error " << hub.getErr() ; 

    // hub calls configure

    return 0 ; 
}
