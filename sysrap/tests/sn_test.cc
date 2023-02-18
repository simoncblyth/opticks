// ./sn_test.sh

/**

https://stackoverflow.com/questions/77005/how-to-automatically-generate-a-stacktrace-when-my-program-crashes

**/


#include <iostream>
#include <iomanip>
#include <cassert>
#include "OpticksCSG.h"
#include "ssys.h"

#include "NP.hh"

const char* FOLD = getenv("FOLD"); 


#include "sn.h"
sn::POOL sn::pool = {} ; 


void test_BinaryTreeHeight()
{
    std::cout << "[ test_BinaryTreeHeight "  << std::endl ; 
    for(int i=0 ; i < 512 ; i++)
    {
        int h0 = sn::BinaryTreeHeight(i) ; 
        int h1 = sn::BinaryTreeHeight_1(i) ; 

        if(sn::level() > 2) std::cout 
           << " i " << std::setw(5) << i 
           << " h0 " << std::setw(5) << h0
           << " h1 " << std::setw(5) << h1
           << std::endl 
           ; 
        assert( h0 == h1 ); 
    } 
    std::cout << "] test_BinaryTreeHeight "  << std::endl ; 
}

void test_ZeroTree()
{
    int num_leaves = 8 ; 
    std::cout << "[ test_ZeroTree num_leaves " << num_leaves << std::endl ; 

    int oper = 1 ; 
    sn* root = sn::ZeroTree(num_leaves, oper ); 
    std::cout << root->render(5) ; 

    if(sn::level() > 1) std::cout << sn::Desc(); 
    if(!sn::LEAK) delete root ;
    if(sn::level() > 1) std::cout << sn::Desc(); 

    std::cout << "] test_ZeroTree num_leaves " << num_leaves << std::endl ; 
}

void test_CommonTree(int num_leaves)
{
    if(sn::level() > 1) std::cout << "[test_CommonTree num_leaves " << num_leaves << std::endl ; 
    if(sn::level() > 1) std::cout << sn::Desc(); 

    std::vector<int> leaftypes ; 
    for(int t=0 ; t < num_leaves ; t++) leaftypes.push_back( CSG_LEAF+t ); 

    sn* root = sn::CommonTree(leaftypes, 1 ); 
    if(sn::level() > 1) std::cout << sn::Desc("CommonTree"); 

    if(sn::level() > -1) std::cout << "test_CommonTree num_leaves " << std::setw(2) << num_leaves << " root: " << root->desc() << std::endl ; 

    if(!sn::LEAK) delete root ; 

    if(sn::level() > 0) std::cout << sn::Desc(); 
    if(sn::level() > 1) std::cout << "]test_CommonTree num_leaves " << num_leaves << std::endl ; 
}

void test_CommonTree()
{
    std::cout << "[ test_CommonTree " << std::endl ; 
    int N = 32 ; 
    for(int nl=1 ; nl < N ; nl++) test_CommonTree(nl); 
    std::cout << "] test_CommonTree " << std::endl ; 
}

sn* manual_tree_0()
{
    sn* l = sn::Prim(CSG_SPHERE); 
    sn* r = sn::Prim(CSG_BOX3); 
    sn* b = sn::Create(CSG_DIFFERENCE, l, r ); 
    return b ; 
}

sn* manual_tree_1()
{
    sn* l = sn::Prim(CSG_SPHERE); 
    sn* r = sn::Prim(CSG_BOX3); 
    sn* b = sn::Create(CSG_UNION, l, r ); 
    return b ; 
}
sn* manual_tree_2()
{
    sn* l = manual_tree_0() ; 
    sn* r = manual_tree_1() ;  
    sn* b = sn::Create(CSG_UNION, l, r ); 
    return b ; 
}
sn* manual_tree_3()
{
    sn* l = manual_tree_0() ; 
    sn* r = sn::Prim(CSG_BOX3);   
    sn* b = sn::Create(CSG_UNION, l, r ); 
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
    std::cout << "[ test_label it " << it  << std::endl ; 

    sn* t = manual_tree(it); 

    t->label(); 

    std::cout << t->render(3) ; 

    if(!sn::LEAK) delete t ; 
    std::cout << "] test_label it " << it  << std::endl ; 
}

void test_positivize()
{
    int it = ssys::getenvint("TREE", 3) ; 
    std::cout << "[ test_positivize it " << it  << std::endl ; 

    sn* t = manual_tree(it); 

    int mode = ssys::getenvint("MODE", 4) ; 

    t->label(); 
    if(sn::level() > 1) std::cout << t->render(mode) ; 

    t->positivize(); 
    std::cout << t->render(mode) ; 

    if(!sn::LEAK) delete t ; 
    std::cout << "] test_positivize it " << it  << std::endl ; 
}

void test_pool()
{
    std::cout << "[ test_pool " << std::endl ; 
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

    std::cout << "] test_pool " << std::endl ; 
}

void test_Simple()
{
    int it = 3 ; 
    std::cout << "[ test_Simple it " << it << std::endl ; 

    sn* t = manual_tree(it); 

    t->label(); 

    std::cout << t->render(5) ; 

    std::cout << sn::Desc() ; 

    if(!sn::LEAK) delete t ; 
    std::cout << "] test_Simple it " << it << std::endl ; 
}

void test_set_left()
{
    std::cout << "[ test_set_left" << std::endl ; 

    sn* l = sn::Prim(CSG_SPHERE); 
    sn* r = sn::Prim(CSG_BOX3); 
    sn* b = sn::Create(CSG_UNION, l, r ); 

    sn* al = sn::Prim(CSG_BOX3) ; 

    b->set_left(al, false); 

    std::cout << sn::Desc() ; 

    delete b ; 
    std::cout << sn::Desc() ; 

    std::cout << "] test_set_left" << std::endl ; 
}


void test_Serialize()
{
    int it = 3 ; 
    std::cout << "[ test_Serialize it " << it  << std::endl ; 

    sn* t = manual_tree(it); 
    std::cout << t->render(5) ; 
    std::cout << sn::Desc(); 
 
    std::vector<_sn> buf ; 
    sn::pool.serialize(buf); 

    delete t ; 

    NP* a = NP::Make<int>( buf.size(), _sn::NV ) ; 
    a->read2<int>((int*)buf.data()); 

    std::cout << " save to " << FOLD << std::endl ; 
    a->save(FOLD, "sn.npy"); 

    std::cout << "] test_Serialize buf.size() " << buf.size()  << std::endl ; 
}

void test_Import()
{
    std::cout << "[ test_Import " << std::endl ; 

    NP* a = NP::Load(FOLD, "sn.npy");
    assert( a->shape[1] == _sn::NV );  
    std::vector<_sn> buf(a->shape[0]) ; 
    a->write<int>((int*)buf.data()); 

    sn::pool.import(buf); 

    int num_root = sn::pool.get_num_root() ; 
    std::cout << " num_root " << num_root << std::endl ; 

    sn* root = sn::pool.get_root(0) ; 
    if(root) std::cout << root->render(5); 
    delete root ; 

    std::cout << "] test_Import " << std::endl ; 
}



int main(int argc, char** argv)
{
    /*
    */
    test_BinaryTreeHeight(); 
    std::cout << sn::pool.brief("0") << std::endl ; 
    test_ZeroTree(); 
    std::cout << sn::pool.brief("1") << std::endl ; 
    test_CommonTree(); 
    std::cout << sn::pool.brief("2") << std::endl ; 
    //test_CommonTree(7); 
    test_label(); 
    std::cout << sn::pool.brief("3") << std::endl ; 
    test_positivize(); 
    std::cout << sn::pool.brief("4") << std::endl ; 
    test_Simple(); 
    std::cout << sn::pool.brief("5") << std::endl ; 
    test_set_left(); 
    std::cout << sn::pool.brief("6") << std::endl ; 
    test_pool(); 
    std::cout << sn::pool.brief("7") << std::endl ; 

    test_Serialize(); 
    std::cout << sn::pool.brief("8") << std::endl ; 
    test_Import(); 
    std::cout << sn::pool.brief("9") << std::endl ; 

    return 0 ; 
}

