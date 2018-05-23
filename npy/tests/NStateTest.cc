#include "NState.hpp"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    for(unsigned i=0 ; i < 16 ; i++)
        LOG(info)
              << std::setw(4) << i 
              << std::setw(4) << NState::FormName(i) 
              ;
              
  
    NState* state = new NState("$HOME/.opticks/rainbow/State", "state" );
    state->setVerbose();
    state->Summary();
    state->load();


}
