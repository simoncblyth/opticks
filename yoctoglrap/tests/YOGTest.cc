#include "OPTICKS_LOG.hh"
#include "BStr.hh"
#include "NGLM.hpp"

#include "YOG.hh"

using YOG::Sc ; 
using YOG::Nd ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    Sc sc ; 
    LOG(info) << sc.desc() ; 

    for(int i=0 ; i < 10 ; i++)
    {
        int ndIdx = sc.add_test_node(i);
        assert( ndIdx == i ); 
        Nd* ndback = sc.nodes.back() ;
        std::cout << ndback->desc() << std::endl ; 
    }    

    LOG(info) << sc.desc() ; 

    return 0 ; 
}


