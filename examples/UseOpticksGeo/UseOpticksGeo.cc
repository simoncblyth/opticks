// opticksgeo/tests/OpticksGeometryTest.cc
// op --topticksgeometry

#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksGeometry.hh"

#include "PLOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "OKGEO_LOG.hh"
#include "GGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_COLOR(argc, argv);

    NPY_LOG__ ;
    OKCORE_LOG__ ;
    OKGEO_LOG__ ;
    GGEO_LOG__ ;

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);      // hub calls configure


    return 0 ; 
}
