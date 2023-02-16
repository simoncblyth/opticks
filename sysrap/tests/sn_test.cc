// ./sn_test.sh

#include <iostream>
#include <iomanip>
#include <cassert>
#include "OpticksCSG.h"

#include "ssys.h"
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
    for(int t=0 ; t < num_leaves ; t++) leaftypes.push_back( CSG_LEAF+t ); 
    sn* root = sn::CommonTree(leaftypes, 1 ); 
    std::cout << root->render() ; 
}

void test_CommonTree_1()
{
    for(int nl=1 ; nl < 32 ; nl++) test_CommonTree_1(nl); 
}

sn* manual_tree_0()
{
    sn* l = sn::Prim(CSG_SPHERE); 
    sn* r = sn::Prim(CSG_BOX3); 
    sn* b = sn::Boolean(CSG_DIFFERENCE, l, r ); 
    return b ; 
}

sn* manual_tree_1()
{
    sn* l = sn::Prim(CSG_SPHERE); 
    sn* r = sn::Prim(CSG_BOX3); 
    sn* b = sn::Boolean(CSG_UNION, l, r ); 
    return b ; 
}

sn* manual_tree_2()
{
    sn* l = manual_tree_0() ; 
    sn* r = manual_tree_1() ;  
    sn* b = sn::Boolean(CSG_UNION, l, r ); 
    return b ; 
}

sn* manual_tree_3()
{
    sn* l = manual_tree_0() ; 
    sn* r = sn::Prim(CSG_BOX3);   
    sn* b = sn::Boolean(CSG_UNION, l, r ); 
    return b ; 
}


sn* manual_tree(int it)
{
    sn* t = nullptr ; 
    switch(it)
    {
        case 0: t = manual_tree_0() ; break ;   
        case 1: t = manual_tree_1() ; break ;   
        case 2: t = manual_tree_2() ; break ;   
        case 3: t = manual_tree_3() ; break ;   
    }
    assert(t); 
    return t ; 
}


void test_label()
{
    int it = 3 ; 
    std::cout << "test_label it " << it  << std::endl ; 

    sn* t = manual_tree(it); 

    t->label(); 

    std::cout << t->render() ; 
}

void test_positivize()
{
    int it = ssys::getenvint("TREE", 3) ; 
    std::cout << "test_positivize it " << it  << std::endl ; 

    sn* t = manual_tree(it); 

    int mode = ssys::getenvint("MODE", 4) ; 

    t->label(); 
    std::cout << t->render(mode) ; 

    t->positivize(); 
    std::cout << t->render(mode) ; 
}




int main(int argc, char** argv)
{
    /*
    test_BinaryTreeHeight(); 
    test_CommonTree_0(); 
    test_CommonTree_1(); 
    test_label(); 
    */

    test_positivize(); 

    return 0 ; 
}

