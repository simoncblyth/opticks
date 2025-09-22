#include "SEvt.hh"

int main()
{
    SEvt* sev = SEvt::Create_EGPU();

    sev->setIndex(0); // needed for config of num photon in the genstep to work

    NP* gs = sev->createInputGenstep_configured();
    std::cout << "gs: " << ( gs ? gs->sstr() : "-" ) << "\n" ;

    std::cout << "gs.repr<int>\n"   << ( gs ? gs->repr<int>()   : "-" ) << "\n" ;
    std::cout << "gs.repr<float>\n" << ( gs ? gs->repr<float>() : "-" ) << "\n" ;


    return 0;
}
