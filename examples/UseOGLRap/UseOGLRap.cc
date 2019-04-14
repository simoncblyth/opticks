// oglrap/tests/AxisAppCheck.cc
#include "AxisApp.hh"
#include "Opticks.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 

    Opticks ok(argc, argv);

    AxisApp aa(&ok); 

    aa.renderLoop();

    return 0 ; 
}



