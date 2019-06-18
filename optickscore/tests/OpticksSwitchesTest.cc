#include "OpticksSwitches.h"

#include "OPTICKS_LOG.hh" 

int main(int argc, char** argv)
{ 
    OPTICKS_LOG(argc, argv); 

    std::string switches = OpticksSwitches(); 

    LOG(info) << switches ; 

    return 0 ; 
}
