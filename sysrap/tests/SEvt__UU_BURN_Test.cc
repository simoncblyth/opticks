
#include "OPTICKS_LOG.hh"
#include "SEvt.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
 
    NP* a = SEvt::UU_BURN ; 
    std::cout << ( a ? a->sstr() : "-" )  << std::endl ; 
    if(a == nullptr) return 0 ; 

    std::cout << a->repr<int>() << std::endl ;  

    int u_idx = 0 ; 
    int ret = a->ifind2D<int>(u_idx, 0, 1 ) ; 

    std::cout << " ret " << ret << std::endl; 



    return 0 ; 
}
