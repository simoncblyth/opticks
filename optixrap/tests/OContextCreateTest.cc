
#include "OXPPNS.hh"
#include "OPTICKS_LOG.hh"

#include "OContext.hh"
#include "Opticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv);  
    ok.configure(); 

    OContext* ctx = OContext::Create( &ok ); 
    delete ctx ; 


    return 0 ; 
}


