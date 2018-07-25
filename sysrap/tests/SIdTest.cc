#include "SId.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
   
    SId* id = new SId("abcdefghijklmnopqrstuvwxyz");
    for(unsigned i=0 ; i < 26 ; i++) std::cout << id->get() << " " ; 
 
    return 0 ; 
}
