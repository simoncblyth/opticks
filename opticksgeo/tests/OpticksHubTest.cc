
#include "Opticks.hh"
#include "OpticksHub.hh"

#include "GGeoTest.hh"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    //const char* funcname = "tboolean-torus--" ;
    const char* funcname = "tboolean-media--" ;
    //const char* funcname = "tboolean-nonexisting--" ;

    Opticks ok(argc, argv, GGeoTest::MakeArgForce(funcname, "--dbgsurf --dbgbnd") );

    OpticksHub hub(&ok);      

    if(hub.getErr()) LOG(fatal) << "hub error " << hub.getErr() ; 

    // hub calls configure

    return 0 ; 
}
