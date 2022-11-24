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

    SEvt* evt = SEvt::CreateOrLoad(); 
    if(evt == nullptr) return 0 ; 
    LOG(info) << evt->desc() ; 
    LOG(info) << evt->descComponent() ; 

    const NP* g4state = evt->getG4State(); 
    LOG(info) << " SEvt::getG4State " << ( g4state ? g4state->sstr() : "-" ); 


    if(SEventConfig::_G4StateRerun > -1) 
    {
        std::cout 
            << "SEventConfig::_G4StateRerun " << SEventConfig::_G4StateRerun 
            << std::endl 
            << g4state->sliceArrayString<unsigned long>( SEventConfig::_G4StateRerun, -1 ) 
            << std::endl 
            ;  
    }

    if(evt->is_loaded)
    {
        evt->setReldir("SEvtLoadTest"); 
        evt->clear_partial("g4state"); 
        evt->save(); 
    }


    return 0 ; 
}
