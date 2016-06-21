#include "NState.hpp"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
  
    NState* state = new NState("$HOME/.opticks/rainbow/State", "state" );
    state->setVerbose();
    state->Summary();
    state->load();


}
