#include "OPTICKS_LOG.hh"
#include "BStr.hh"
#include "NGLM.hpp"

#include "YOG.hh"

using YOG::Sc ; 
using YOG::Nd ; 



int test_add_node( Sc& sc, int idx)
{
    int lvIdx = idx ; 
    std::string lvName = BStr::concat<int>("lv", idx, NULL) ;   
    std::string pvName = BStr::concat<int>("pv", idx, NULL) ;   
    std::string soName = BStr::concat<int>("so", idx, NULL) ;   
    const glm::mat4* transform = new glm::mat4 ; 
    std::string boundary = BStr::concat<int>("bd", idx, NULL) ;   
    int depth = 0 ; 
    bool selected = true ;  

    int ndIdx = sc.add_node(lvIdx, 
                            lvName, 
                            pvName, 
                            soName, 
                            transform, 
                            boundary,
                            depth, 
                            selected);  

    return ndIdx ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    Sc sc ; 
    LOG(info) << sc.desc() ; 

    for(int i=0 ; i < 100 ; i++)
    {
        int ndIdx = test_add_node(sc, i);
        assert( ndIdx == i ); 
        Nd* ndback = sc.nodes.back() ;
        std::cout << ndback->desc() << std::endl ; 
    }    

    LOG(info) << sc.desc() ; 

    return 0 ; 
}


