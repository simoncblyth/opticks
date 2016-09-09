// op --topticksgeometry

#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksGeometry.hh"

#include "PLOG.hh"
#include "OKCORE_LOG.hh"
#include "OKGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    OKCORE_LOG_ ;
    OKGEO_LOG_ ;

    Opticks ok(argc, argv);

    OpticksHub hub(&ok);


    return 0 ; 
}
