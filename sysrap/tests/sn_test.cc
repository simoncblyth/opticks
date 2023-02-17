// ./sn_test.sh

/**

https://stackoverflow.com/questions/77005/how-to-automatically-generate-a-stacktrace-when-my-program-crashes

**/


#include <iostream>
#include <iomanip>
#include <cassert>
#include "OpticksCSG.h"

#include "ssys.h"

#include "sn.h"
std::map<int, sn*> sn::pool = {} ; 
int sn::count = 0 ; 


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

void test_ZeroTree()
{
    int num_leaves = 8 ; 
    std::cout << "test_ZeroTree num_leaves " << num_leaves << std::endl ; 

    int oper = 1 ; 
    sn* root = sn::ZeroTree(num_leaves, oper ); 
    std::cout << root->render() ; 

    std::cout << sn::Desc(); 
    if(!sn::LEAK) delete root ;
    std::cout << sn::Desc(); 

}

void test_CommonTree(int num_leaves)
{
    std::cout << "test_CommonTree num_leaves " << num_leaves << std::endl ; 
    std::cout << sn::Desc(); 


    std::vector<int> leaftypes ; 
    for(int t=0 ; t < num_leaves ; t++) leaftypes.push_back( CSG_LEAF+t ); 

    std::cout << "[sn::CommonTree" << std::endl ; 
    sn* root = sn::CommonTree(leaftypes, 1 ); 
    std::cout << "]sn::CommonTree" << std::endl ; 

    std::cout << sn::Desc(); 

    std::cout << " root->desc " << root->desc() << std::endl ; 

    if(!sn::LEAK) 
    {
        std::cout << "[ delete root  root->pid " << root->pid << std::endl ; 
        delete root ;
        std::cout << "]" << std::endl ; 
    }
    else
    {
        std::cout << " root leaked " << std::endl;  
    }

    std::cout << "[" << std::endl ; 
    std::cout << sn::Desc(); 
    std::cout << "]" << std::endl ; 

}

void test_CommonTree()
{
    int N = 32 ; 
    for(int nl=1 ; nl < N ; nl++) test_CommonTree(nl); 
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

void test_pool()
{
    assert( sn::pool.size() == 0  );  
    sn* a = sn::Zero() ; 


    assert( sn::pool.size() == 1  );  
    sn* b = sn::Zero() ; 

    assert( sn::pool.size() == 2  );  
    sn* c = sn::Zero() ; 

    assert( sn::pool.size() == 3  );  


    std::cout << sn::Desc() ; 


    delete c ; 
    assert( sn::pool.size() == 2  );  

    delete a ; 
    assert( sn::pool.size() == 1  );  

    delete b ; 
    assert( sn::pool.size() == 0  );  
}

void test_Simple()
{
    int it = 3 ; 
    std::cout << "test_Simple it " << it << std::endl ; 

    sn* t = manual_tree(it); 

    t->label(); 

    std::cout << t->render() ; 

    std::cout << sn::Desc() ; 

}

void test_set_left()
{
    std::cout << "test_set_left" << std::endl ; 

    sn* l = sn::Prim(CSG_SPHERE); 
    sn* r = sn::Prim(CSG_BOX3); 
    sn* b = sn::Boolean(CSG_UNION, l, r ); 

    sn* al = sn::Prim(CSG_BOX3) ; 

    b->set_left(al, false); 

    std::cout << sn::Desc() ; 


    delete b ; 
    std::cout << sn::Desc() ; 
}





int main(int argc, char** argv)
{
    /*
    test_BinaryTreeHeight(); 
    test_ZeroTree(); 
    test_CommonTree(); 
    test_label(); 
    test_positivize(); 
    test_pool(); 
    */

    //test_CommonTree(1);  
    //test_CommonTree(2);  
    //test_CommonTree(3);  
    //test_CommonTree(4); 
    //test_CommonTree(5); 
    test_CommonTree(6); 
    //test_CommonTree(7); 
    //test_CommonTree(8); 

    //test_Simple(); 
    //test_set_left(); 

    return 0 ; 
}

