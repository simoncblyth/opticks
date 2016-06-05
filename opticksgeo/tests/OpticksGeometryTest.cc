// op --topticksgeometry

#include "Opticks.hh"
#include "OpticksGeometry.hh"

int main(int argc, char** argv)
{
    Opticks ok(argc, argv, "OpticksGeometry.log");

    OpticksGeometry og(&ok);

    og.loadGeometry();

    return 0 ; 
}
