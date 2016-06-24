// op --topticksgeometry

#include "Opticks.hh"
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

    ok.configure();

    OpticksGeometry og(&ok);

    og.loadGeometry();

    return 0 ; 
}
