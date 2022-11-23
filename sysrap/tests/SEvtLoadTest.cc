/**

    epsilon:tests blyth$ GEOM=hamaLogicalPMT SOpticksResource_ExecutableName=OverrideExecutableName SEvtLoadTest 
    U::DirList path /tmp/blyth/opticks/GEOM/hamaLogicalPMT/OverrideExecutableName/ALL ext - NO ENTRIES FOUND 
    sevent::descMax    


**/

#include "OPTICKS_LOG.hh"
#include <iostream>
#include "SEvt.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ; 

    SEvt* sev = SEvt::CreateOrLoad(); 
    if(sev == nullptr) return 0 ; 
    LOG(info) << sev->desc() ; 
    LOG(info) << sev->descComponent() ; 

    const NP* g4state = sev->getG4State(); 
    LOG(info) << " SEvt::getG4State " << ( g4state ? g4state->sstr() : "-" ); 


    return 0 ; 
}
