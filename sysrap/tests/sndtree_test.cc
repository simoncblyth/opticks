/**
sndtree_test.cc
=================


snd::render width 7 height 2
            .                   
                                
    .               .           
                                
.       .       .       .       
                                

 snd::Inorder [ 4 6 5 10 7 9 8 ]

Notice that the node indices here all exceed the spheres 0,1,2,3 
as the operator nodes and CSG_ZERO nodes are all created afterwards. 

Perhaps can avoid wasting node indices by directly popping the leaves
and using them during the build ? 

**/

#include <iostream>
#include "OpticksCSG.h"
#include "snd.hh"
#include "sndtree.h"

#include "stree.h"

int main(int argc, char** argv)
{
    stree st ; 

    int a = snd::Sphere(100.) ; 
    int b = snd::Sphere(100.) ; 
    int c = snd::Sphere(100.) ; 
    int d = snd::Sphere(100.) ; 

    std::vector<int> leaves = {a,b,c,d} ; 

    int root = sndtree::CommonTree( leaves, CSG_UNION ); 
    snd::SetLVID(root, 0 );  
    // without snd::SetLVID depth is not set : so all nodes render at same depth
 
    std::cout << "root " << root << std::endl ;  
    std::cout << "snd::Render(root) " << snd::Render(root) << std::endl ; 


    std::vector<int> order ; 
    snd::Inorder(order, root ); 

    std::cout << " snd::Inorder [ " ; 
    for(int i=0 ; i < int(order.size()) ; i++) std::cout << order[i] << " " ; 
    std::cout << "]" << std::endl ; 

    //  snd::Inorder [ 4 6 5 10 7 9 8 ]

    return 0 ; 
}
