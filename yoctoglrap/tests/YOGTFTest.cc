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
    OPTICKS_LOG(argc, argv);

    Sc sc ; 

    LOG(info) << sc.desc() ; 

    int N = 3 ; 
    for(int i=0 ; i < N ; i++)
    {
        int ndIdx = sc.add_test_node(i);
        std::string lvName = "lvName" ; 
        std::string soName = "soName" ; 
        sc.add_mesh(i, lvName, soName );  

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


