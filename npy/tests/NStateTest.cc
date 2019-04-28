#include "NState.hpp"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 

    for(unsigned i=0 ; i < 16 ; i++)
        LOG(info)
              << std::setw(4) << i 
              << std::setw(4) << NState::FormName(i) 
              ;
              
  
    NState* state = new NState("$HOME/.opticks/rainbow/State", "state" );
    state->setVerbose();
    state->Summary();
    state->load();

    return 0 ; 
}
