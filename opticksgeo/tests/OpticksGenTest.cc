
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "OpticksGen.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 


    



    return 0 ; 
}
