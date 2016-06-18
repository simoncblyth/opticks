#include "BLog.hh"
#include "NState.hpp"

int main(int argc, char** argv)
{
    BLOG(argc, argv);
  
    NState* state = new NState("$HOME/.opticks/rainbow/State", "state" );
    state->setVerbose();
    state->Summary();
    state->load();


}
