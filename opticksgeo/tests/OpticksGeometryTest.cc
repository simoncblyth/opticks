// op --topticksgeometry

#include "Opticks.hh"
#include "GCache.hh"
#include "OpticksGeometry.hh"


int main(int argc, char** argv)
{
    Opticks ok(argc, argv, "OpticksGeometry.log");

    GCache gc(&ok) ;

    OpticksGeometry og(&ok, &gc);

    og.loadGeometry();


    return 0 ; 
}
