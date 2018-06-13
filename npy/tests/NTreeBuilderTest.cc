#include <cstdlib>
#include <cassert>
#include <vector>

#include "OPTICKS_LOG.hh"

#include "NBox.hpp"
#include "NNode.hpp"
#include "NTreeBuilder.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv); 

    unsigned nprim = argc > 1 ? atoi(argv[1]) : 6  ; 

    std::vector<nnode*> prims ; 
    for(unsigned i=0 ; i < nprim ; i++)
    {
        nnode* a = new nbox(make_box3(400,400,400));
        prims.push_back(a);  
    }

    nnode* root = NTreeBuilder<nnode>::UnionTree(prims) ; 
    assert( root ) ; 
    //root->dump();

    return 0 ; 
}

/*

2018-06-13 19:37:56.887 INFO  [19092658] [NTreeBuilder<nnode>::analyse@93]  NNodeAnalyse 
NNodeAnalyse height 3 count 15
                              un                            

              un                              un            

      un              un              un              un    

  ze      ze      ze      ze      ze      ze      ze      ze


2018-06-13 19:37:56.888 INFO  [19092658] [NTreeBuilder<nnode>::analyse@93]  NNodeAnalyse 
NNodeAnalyse height 3 count 15
                              un                            

              un                              un            

      un              un              un              un    

  bo      bo      bo      bo      bo      ze      ze      ze


2018-06-13 19:37:56.888 INFO  [19092658] [NTreeBuilder<nnode>::analyse@93]  NNodeAnalyse 
NNodeAnalyse height 3 count 9
                              un    

              un                  bo

      un              un            

  bo      bo      bo      bo        


2018-06-13 19:37:56.888 INFO  [19092658] [*NTreeBuilder<nnode>::CommonTree@19]  num_prims 5 height 3 operator union

*/
