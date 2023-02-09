/**
sndtree_test.cc
=================

::

   ./sndtree_test.sh 

**/

#include <iostream>
#include "OpticksCSG.h"
#include "snd.hh"
#include "sndtree.h"

#include "stree.h"

void test_Inorder(int root)
{
    std::vector<int> order ; 
    snd::Inorder(order, root ); 

    std::cout << " snd::Inorder [ " ; 
    for(int i=0 ; i < int(order.size()) ; i++) std::cout << order[i] << " " ; 
    std::cout << "]" << std::endl ; 
}



int main(int argc, char** argv)
{
    stree st ; 

    int a = snd::Sphere(100.) ; 
    int b = snd::Sphere(100.) ; 
    int c = snd::Sphere(100.) ; 
    int d = snd::Sphere(100.) ; 

    std::vector<int> leaves = {a,b,c,d} ; 

    int root = sndtree::CommonTree( leaves, CSG_UNION ); 
 
    std::cout << "root " << root << std::endl ;  
    std::cout << "snd::Render(root) " << snd::Render(root) << std::endl ; 


    test_Inorder(root); 


    return 0 ; 
}
