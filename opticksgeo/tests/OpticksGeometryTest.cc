// op --topticksgeometry

#include "Opticks.hh"
#include "OpticksGeometry.hh"
#include "BSys.hh"

int main(int argc, char** argv)
{
    Opticks ok(argc, argv, "OpticksGeometry.log");

    ok.configure();

    OpticksGeometry og(&ok);

    og.loadGeometry();

    BSys::WaitForInput("OpticksGeometryTest::main waiting..."); 

    return 0 ; 
}
