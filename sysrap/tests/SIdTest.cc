#include "SId.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
   
    SId* id = new SId("abcdef");


    
    for(unsigned i=0 ; i < 24 ; i++) std::cout << id->get() << " " ; 
    std::cout << std::endl ;  

    id->reset(); 
    for(unsigned i=0 ; i < 24 ; i++) std::cout << id->get() << " " ; 
    std::cout << std::endl ;  


 
    return 0 ; 
}
