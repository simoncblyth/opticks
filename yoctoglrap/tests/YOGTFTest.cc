#include "OPTICKS_LOG.hh"
#include "BStr.hh"
#include "NGLM.hpp"

#include "YOG.hh"
#include "YOGTF.hh"

using YOG::Sc ; 
using YOG::Nd ; 

using YOG::TF ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    Sc sc ; 

    LOG(info) << sc.desc() ; 
    for(int i=0 ; i < 3 ; i++)
    {
        int ndIdx = sc.add_test_node(i);
        assert( ndIdx == i ); 
        Nd* nd = sc.nodes.back() ;
        nd->children = { i+1, i+2 } ;  // purely dummy  

        std::cout << nd->desc() << std::endl ; 
    }    
    LOG(info) << sc.desc() ; 


    TF tf(&sc); 

    const char* path = "/tmp/YOGTFTest.gltf" ; 
    tf.save(path);     


    return 0 ; 
}


