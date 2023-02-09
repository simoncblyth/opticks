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

void test_CommonTree_0()
{
    int num_leaves = 8 ; 
    int oper = 1 ; 
    sn* root = sn::CommonTree(num_leaves, oper ); 
    std::cout << root->render() ; 
}

void test_CommonTree_1(int num_leaves)
{
    std::cout << "test_CommonTree_1 num_leaves " << num_leaves << std::endl ; 
    std::vector<int> leaftypes ; 
    for(int t=0 ; t < num_leaves ; t++) leaftypes.push_back( 100+t ); 
    sn* root = sn::CommonTree(leaftypes, 1 ); 
    std::cout << root->render() ; 
}

void test_CommonTree_1()
{
    for(int nl=1 ; nl < 32 ; nl++) test_CommonTree_1(nl); 
}



int main(int argc, char** argv)
{
    /*
    test_BinaryTreeHeight(); 
    test_CommonTree_0(); 
    test_CommonTree_1(); 
    */
    test_CommonTree_1(); 


    return 0 ; 
}

