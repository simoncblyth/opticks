// TEST=OpticksGeoTest om-t

#include "Opticks.hh"
#include "OpticksHub.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);      // hub calls configure

    return 0 ; 
}
