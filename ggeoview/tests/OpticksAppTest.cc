#include "Opticks.hh"
#include "OpticksApp.hh"
#include "OpticksEvent.hh"

#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    Opticks* opticks = new Opticks(argc, argv);

    opticks->configure();

    OpticksApp* app = new OpticksApp( opticks );

    OpticksEvent* evt = opticks->makeEvent();   




    return 0 ; 
}

