/**
SEvt__createInputGenstep_configuredTest.cc
===========================================

Usage::

    ~/o/sysrap/tests/SEvt__createInputGenstep_configuredTest.sh
    ~/o/sysrap/tests/SEvt__createInputGenstep_configuredTest_SML.sh

**/

#include "SEvt.hh"
#include "ssys.h"

int main()
{
    SEvt* sev = SEvt::Create_EGPU();

    sev->setIndex(0); // needed for config of num photon in the genstep to work

    const char* GS_NAME = ssys::getenvvar("GS_NAME", "gs.npy");

    NP* gs = sev->createInputGenstep_configured();

    std::cout << "gs: " << ( gs ? gs->sstr() : "-" ) << " GS_NAME[" << GS_NAME << "]\n" ;
    std::cout << "gs.repr<int>\n"   << ( gs ? gs->repr<int>()   : "-" ) << "\n" ;
    std::cout << "gs.repr<float>\n" << ( gs ? gs->repr<float>() : "-" ) << "\n" ;

    gs->save("$FOLD/$GS_NAME");

    return 0;
}
