// name=sn_test ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>
#include <cassert>

#include "sn.h"

void test_BinaryTreeHeight()
{
    for(int i=0 ; i < 512 ; i++)
    {
        int h0 = sn::BinaryTreeHeight(i) ; 
        int h1 = sn::BinaryTreeHeight_1(i) ; 

        std::cout 
           << " i " << std::setw(5) << i 
           << " h0 " << std::setw(5) << h0
           << " h1 " << std::setw(5) << h1
           << std::endl 
           ; 
        assert( h0 == h1 ); 
    } 
}

void test_CommonTree()
{
    int num_leaves = 8 ; 
    int op = 1 ; 

    sn* root = sn::CommonTree(num_leaves, op ); 

    std::cout << root->render() ; 
}



int main(int argc, char** argv)
{
    /*
    test_BinaryTreeHeight(); 
    */

    test_CommonTree(); 

    return 0 ; 
}

