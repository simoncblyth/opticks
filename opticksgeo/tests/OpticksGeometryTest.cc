// op --topticksgeometry

#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksGeometry.hh"

#include "PLOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "OKGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NPY_LOG__ ;
    OKCORE_LOG__ ;
    OKGEO_LOG__ ;

    Opticks ok(argc, argv);

    OpticksHub hub(&ok);


    return 0 ; 
}
