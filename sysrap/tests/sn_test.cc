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

void check(const char* msg, int i=-1)
{
    //const char* spacer = "\n\n" ; 
    const char* spacer = "" ; 

    std::stringstream ss ; 
    ss << "check[" << std::setw(3) << i << "] " << std::setw(30) << ( msg ? msg : "-" ) << "  " << sn::pool.brief() << spacer << std::endl ; 
    std::string str = ss.str(); 
    std::cout << str ; 

    if(!sn::LEAK)
    {
        assert( sn::pool.size() == 0 ); 
    }
}

void test_BinaryTreeHeight()
{
    if(sn::level() > 0) std::cout << "[ test_BinaryTreeHeight "  << std::endl ; 
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
    if(sn::level() > 0) std::cout << "] test_BinaryTreeHeight "  << std::endl ; 

    check("BinaryTreeHeight"); 
}

void test_ZeroTree()
{
    int num_leaves = 8 ; 
    if(sn::level() > 0 ) std::cout << "[ test_ZeroTree num_leaves " << num_leaves << std::endl ; 

    int oper = 1 ; 
    sn* root = sn::ZeroTree(num_leaves, oper ); 

    if(sn::level() > 1) std::cout << root->render(5) ; 
    if(sn::level() > 1) std::cout << root->render(1) ; 

    if(sn::level() > 1) std::cout << sn::Desc(); 
    if(!sn::LEAK) delete root ;
    if(sn::level() > 1) std::cout << sn::Desc(); 

    if(sn::level() > 0) std::cout << "] test_ZeroTree num_leaves " << num_leaves << std::endl ; 
    check("ZeroTree"); 
}

void test_CommonTree(int num_leaves)
{
    if(sn::level() > 1) std::cout << "[test_CommonTree num_leaves " << num_leaves << std::endl ; 
    if(sn::level() > 1) std::cout << sn::Desc(); 

    std::vector<int> leaftypes ; 
    for(int t=0 ; t < num_leaves ; t++) leaftypes.push_back( CSG_LEAF+t ); 

    sn* root = sn::CommonTree(leaftypes, 1 ); 
    if(sn::level() > 1) std::cout << sn::Desc("CommonTree"); 

    if(sn::level() > 0) std::cout << "test_CommonTree num_leaves " << std::setw(2) << num_leaves << " root: " << root->desc() << std::endl ; 

    if(!sn::LEAK) delete root ; 

    if(sn::level() > 0) std::cout << sn::Desc(); 
    if(sn::level() > 1) std::cout << "]test_CommonTree num_leaves " << num_leaves << std::endl ; 
    check("CommonTree", num_leaves); 
}

void test_CommonTree()
{
    if(sn::level() > 0) std::cout << "[ test_CommonTree " << std::endl ; 
    int N = 32 ; 
    for(int nl=1 ; nl < N ; nl++) test_CommonTree(nl); 
    if(sn::level() > 0) std::cout << "] test_CommonTree " << std::endl ; 
    check("CommonTree"); 
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
    if(sn::level() > 0) std::cout << "[ test_label it " << it  << std::endl ; 

    sn* t = manual_tree(it); 

    t->label(); 

    if(sn::level() > 1) std::cout << t->render(3) ; 

    if(!sn::LEAK) delete t ; 
    if(sn::level() > 0) std::cout << "] test_label it " << it  << std::endl ; 
    check("label"); 
}

void test_positivize()
{
    int it = ssys::getenvint("TREE", 3) ; 
    if(sn::level() > 0) std::cout << "[ test_positivize it " << it  << std::endl ; 

    sn* t = manual_tree(it); 

    int mode = ssys::getenvint("MODE", 4) ; 

    t->label(); 
    if(sn::level() > 1) std::cout << t->render(mode) ; 

    t->positivize(); 
    if(sn::level() > 1) std::cout << t->render(mode) ; 

    if(!sn::LEAK) delete t ; 
    if(sn::level() > 0) std::cout << "] test_positivize it " << it  << std::endl ; 
    check("positivize"); 
}

void test_pool()
{
    if(sn::level() > 0) std::cout << "[ test_pool " << std::endl ; 
    assert( sn::pool.size() == 0  );  
    sn* a = sn::Zero() ; 


    assert( sn::pool.size() == 1  );  
    sn* b = sn::Zero() ; 

    assert( sn::pool.size() == 2  );  
    sn* c = sn::Zero() ; 

    assert( sn::pool.size() == 3  );  


    if(sn::level() > 1) std::cout << sn::Desc() ; 


    delete c ; 
    assert( sn::pool.size() == 2  );  

    delete a ; 
    assert( sn::pool.size() == 1  );  

    delete b ; 
    assert( sn::pool.size() == 0  );  

    if(sn::level() > 0) std::cout << "] test_pool " << std::endl ; 
    check("pool"); 
}

void test_Simple()
{
    int it = 3 ; 
    if(sn::level() > 0) std::cout << "[ test_Simple it " << it << std::endl ; 

    sn* t = manual_tree(it); 

    t->label(); 

    if(sn::level() > 1) std::cout << t->render(5) ; 
    if(sn::level() > 1) std::cout << sn::Desc() ; 

    if(!sn::LEAK) delete t ; 
    if(sn::level() > 0) std::cout << "] test_Simple it " << it << std::endl ; 
    check("Simple"); 
}

void test_set_left()
{
    if(sn::level() > 0) std::cout << "[ test_set_left" << std::endl ; 

    sn* l = sn::Prim(CSG_SPHERE); 
    sn* r = sn::Prim(CSG_BOX3); 
    sn* b = sn::Create(CSG_UNION, l, r ); 

    sn* al = sn::Prim(CSG_BOX3) ; 

    b->set_left(al, false); 

    if(sn::level() > 1) std::cout << sn::Desc() ; 

    delete b ; 
    if(sn::level() > 1) std::cout << sn::Desc() ; 

    if(sn::level() > 0) std::cout << "] test_set_left" << std::endl ; 
    check("set_left"); 
}


void test_Serialize()
{
    int lev = 0 ; 
    int it = 3 ; 
    if(sn::level() > lev) std::cout << "[ test_Serialize it " << it  << std::endl ; 

    sn* t = manual_tree(it); 
    if(sn::level() > lev) std::cout << t->render(5) ; 
    if(sn::level() > lev) std::cout << sn::Desc(); 

    int num_root = sn::pool.get_num_root() ; 
    if(sn::level() > lev) std::cout << " num_root " << num_root << std::endl ; 

 
    std::vector<_sn> buf ; 
    sn::pool.serialize(buf); 

    delete t ; 

    NP* a = NP::Make<int>( buf.size(), _sn::NV ) ; 
    a->read2<int>((int*)buf.data()); 

    if(sn::level() > lev) std::cout << " save to " << FOLD << "/" << sn::NAME << std::endl ; 
    a->save(FOLD, sn::NAME); 


    if(sn::level() > lev) std::cout << "] test_Serialize buf.size() " << buf.size()  << std::endl ; 
    if(sn::level() > lev) std::cout << sn::pool.Desc(buf) << std::endl ; 

    check("Serialize"); 
}

void test_Import()
{
    int lev = 0 ; 
    if(sn::level() > lev) std::cout << "[ test_Import " << std::endl ; 

    if(sn::level() > lev) std::cout << " load from " << FOLD << "/" << sn::NAME << std::endl ; 
    NP* a = NP::Load(FOLD, sn::NAME );
    assert( a->shape[1] == _sn::NV );  
    std::vector<_sn> buf(a->shape[0]) ; 
    a->write<int>((int*)buf.data()); 

    sn::pool.import(buf); 


    if(sn::level() > lev) std::cout << sn::pool.Desc(buf) << std::endl ; 

    int num_root = sn::pool.get_num_root() ; 
    if(sn::level() > lev) std::cout << " num_root " << num_root << std::endl ; 
    if(sn::level() > lev) std::cout << sn::Desc(); 


    sn* root = sn::pool.get_root(0) ; 
    if(root && sn::level() > lev) std::cout << root->render(5); 
    delete root ; 

    if(sn::level() > lev) std::cout << "] test_Import " << std::endl ; 
    check("Import"); 
}


void test_dtor_0()
{
    sn* n = sn::Zero(); 
    delete n ; 
    check("dtor_0"); 
}
void test_dtor_1()
{
    sn* a = sn::Prim(100); 
    sn* b = sn::Prim(101);
    sn* c = sn::Create(1, a, b ); 
    delete c ; 
    check("dtor_1"); 
}

void Desc()
{
#ifdef WITH_CHILD
   if(sn::level() > -1) std::cout << "WITH_CHILD " ;  
#else
   if(sn::level() > -1) std::cout << "NOT:WITH_CHILD " ; 
#endif
   std::cout << " level : " << sn::level() << std::endl ; 
}

/**


8 leaves needs no pruning::

                                   1

                  10                               11

           100           101                 110             111

        1000 1001    1010  1011         1100   1101       1110  1111


7 leaves, just needs one bileaf converted to a prim::

                                   1

                  10                               11

           100           101                 110             111

        1000 1001    1010  1011         1100   1101       



6 leaves needs two bileaf converted to prims::


                                   1

                  10                               11

           100           101                 110             111

        1000 1001    1010  1011                                           


**/


void test_set_child()
{
    sn* a = sn::Prim(100); 
    sn* b = sn::Prim(101); 
    sn* c = sn::Create(1, a, b ); 

    if(sn::level() > 0) std::cout << c->render(1) ; 

    sn* a2 = sn::Prim(200); 
    sn* b2 = sn::Prim(201); 

    c->set_child(0, a2, false); 
    c->set_child(1, b2, false); 

    if(sn::level() > 0) std::cout << c->render(1) ; 

    delete c ; 

    check("set_child"); 
}

void test_deepcopy_0()
{
    sn* a = sn::Prim(100); 
    sn* b = sn::Prim(101); 
    sn* c = sn::Create(1, a, b ); 

    if(sn::level() > 0) std::cout << c->render(1) ; 
    if(sn::level() > 0) std::cout << c->desc_r() ; 

    sn* ccp = c->deepcopy(); 

#ifdef WITH_CHILD
    assert( ccp->child.size() == c->child.size() ); 
#endif

    delete c ; 

    if(sn::level() > 0) std::cout << ccp->render(1) ; 
    if(sn::level() > 0) std::cout << ccp->desc_r() ; 

    delete ccp ; 

    check("deepcopy_0"); 
}

void test_deepcopy_1_leaking()
{
    sn* a = sn::Prim(100); 
    sn* b = sn::Prim(101); 
    sn* c = sn::Create(1, a, b ); 

    std::cout << "c\n " << c->desc_child() << std::endl ; 

    sn* cp = new sn(*c) ; 

    // NB the child vector is shallow copied by this default copy ctor
    // which causes the cp and c to both think they own a and b 
    // which will cause ownership isses when delete 
    //
    // hence cannot easily clean up this situation

    std::cout << "cp\n" << cp->desc_child() << std::endl ; 

    //check("deepcopy_1_leaking"); 
}

void test_next_sibling()
{
    int lev = 0 ; 

    sn* a = sn::Prim(100); 
    sn* b = sn::Prim(101); 
    sn* c = sn::Create(1, a, b ); 

    int ia = a->sibling_index() ;  
    int ib = b->sibling_index() ;  

    if(sn::level() > lev) std::cerr << "test_next_sibling ia  " << ia  << std::endl ;     
    if(sn::level() > lev) std::cerr << "test_next_sibling ib  " << ib  << std::endl ;     
    
    const sn* x = a->next_sibling() ; 
    const sn* y = b->next_sibling() ; 
    const sn* z = c->next_sibling() ; 

    if(sn::level() > lev) std::cerr << "test_next_sibling x: " << ( x ? "Y" : "N" ) << std::endl ;     
    if(sn::level() > lev) std::cerr << "test_next_sibling y: " << ( y ? "Y" : "N" ) << std::endl ;     
    if(sn::level() > lev) std::cerr << "test_next_sibling z: " << ( z ? "Y" : "N" ) << std::endl ;     

    assert( x == b ); 
    assert( y == nullptr ); 
    assert( z == nullptr ); 

    delete c ; 

    check("next_sibling"); 
}



int main(int argc, char** argv)
{


    test_next_sibling(); 
    test_deepcopy_0(); 
    //test_deepcopy_1_leaking(); 

    // before fixing sn::deepcopy with sn::disown_child some of the below  segmented or hung
    test_CommonTree(1);
    test_CommonTree(2);
    test_CommonTree(3);
    test_CommonTree(4);
    test_CommonTree(5);
    test_CommonTree(6);  
    test_CommonTree(7); 
    test_CommonTree(8);
    test_CommonTree(9);
    test_CommonTree(10);
    test_CommonTree(11); 
    test_CommonTree(12);
    test_CommonTree(16);   
    test_CommonTree(32); 
    
    test_set_child(); 
    test_CommonTree(4); 
    test_CommonTree(6); 
    test_dtor_0(); 
    test_dtor_1(); 
    test_BinaryTreeHeight(); 
    test_ZeroTree(); 
    test_CommonTree(); 
    test_label(); 
    test_positivize(); 
    test_Simple(); 
    test_set_left(); 
    test_pool(); 

  
    test_Serialize(); 
    test_Import(); 


    Desc(); 

    return 0 ; 
}


