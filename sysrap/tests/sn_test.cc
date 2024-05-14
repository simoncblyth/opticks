/**
~/o/sysrap/tests/sn_test.cc
=============================

::

    ~/o/sysrap/tests/sn_test.sh 


* https://stackoverflow.com/questions/77005/how-to-automatically-generate-a-stacktrace-when-my-program-crashes

**/

#include <iostream>
#include <iomanip>
#include <cassert>
#include "OpticksCSG.h"
#include "ssys.h"
#include "s_csg.h"

#include "NPX.h"

const char* FOLD = getenv("FOLD"); 

#include "sn.h"

void Desc()
{
#ifdef WITH_CHILD
    if(sn::level() > -1) std::cout << "WITH_CHILD " ;  
#else
    if(sn::level() > -1) std::cout << "NOT:WITH_CHILD " ; 
#endif
    std::cout << " level : " << sn::level() << std::endl ; 
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
sn* manual_tree_4()
{
    sn* l = sn::Prim(CSG_SPHERE); 
    sn* r = sn::Prim(CSG_BOX3); 

    glm::tmat4x4<double> t(1.); 
    t[3][2] = 1000. ; 
    r->setXF(t); 

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
        case 4: t = manual_tree_4() ; break ;   
    }
    assert(t); 
    return t ; 
}



struct sn_test
{
    static int check_LEAK(const char* msg, int i=-1);

    static int BinaryTreeHeight();
    static int ZeroTree();
    static int CommonOperatorTypeTree(int num_leaves);
    static int CommonOperatorTypeTree();
    static int label();
    static int positivize();
    static int pool();
    static int Simple();
    static int set_left();
    static int serialize_0();
    static int serialize_1();
    static int import_0();
    static int import_1();
    static int dtor_0();
    static int dtor_1();
    static int set_child();
    static int deepcopy_0();
    static int deepcopy_1_leaking();
    static int next_sibling();
    static int Serialize();
    static int Import();
    static int OrderPrim_();

    static int ALL();
    static int main();
};


int sn_test::check_LEAK(const char* msg, int i)
{
    //const char* spacer = "\n\n" ; 
    const char* spacer = "" ; 

    std::stringstream ss ; 
    ss << "check_LEAK[" << std::setw(3) << i << "] " << std::setw(30) << ( msg ? msg : "-" ) << "  " << sn::pool->brief() << spacer << std::endl ; 
    std::string str = ss.str(); 
    std::cout << str ; 

    if(!sn::LEAK)
    {
        assert( sn::pool->size() == 0 ); 
    }

    if(!s_tv::LEAK)
    {
        assert( s_tv::pool->size() == 0 ); 
    }

    return 0 ; 
}


int sn_test::BinaryTreeHeight()
{
    if(sn::level() > 0) std::cout << "[ sn_test::BinaryTreeHeight "  << std::endl ; 
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
    if(sn::level() > 0) std::cout << "] sn_test::BinaryTreeHeight "  << std::endl ; 

    check_LEAK("BinaryTreeHeight"); 
    return 0 ; 
}

int sn_test::ZeroTree()
{
    int num_leaves = 8 ; 
    if(sn::level() > 0 ) std::cout << "[ sn_test::ZeroTree num_leaves " << num_leaves << std::endl ; 

    int oper = 1 ; 
    sn* root = sn::ZeroTree(num_leaves, oper ); 

    if(sn::level() > 1) std::cout << root->render(5) ; 
    if(sn::level() > 1) std::cout << root->render(1) ; 

    if(sn::level() > 1) std::cout << sn::Desc(); 
    if(!sn::LEAK) delete root ;
    if(sn::level() > 1) std::cout << sn::Desc(); 

    if(sn::level() > 0) std::cout << "] sn_test::ZeroTree num_leaves " << num_leaves << std::endl ; 
    check_LEAK("ZeroTree"); 
    return 0 ; 
}

int sn_test::CommonOperatorTypeTree(int num_leaves)
{
    if(sn::level() > 1) std::cout << "[sn_test::CommonOperatorTypeTree num_leaves " << num_leaves << std::endl ; 
    if(sn::level() > 1) std::cout << sn::Desc(); 

    std::vector<int> leaftypes ; 
    for(int t=0 ; t < num_leaves ; t++) leaftypes.push_back( CSG_LEAF+t ); 

    sn* root = sn::CommonOperatorTypeTree(leaftypes, CSG_UNION ) ; 

    if(sn::level() > 1) std::cout << sn::Desc(root); 

    if(sn::level() > 0) std::cout << "sn_test::CommonOperatorTypeTree num_leaves " << std::setw(2) << num_leaves << " root: " << root->desc() << std::endl ; 

    if(!sn::LEAK) delete root ; 

    if(sn::level() > 0) std::cout << sn::Desc(); 
    if(sn::level() > 1) std::cout << "]sn_test::CommonOperatorTypeTree num_leaves " << num_leaves << std::endl ; 
    check_LEAK("CommonOperatorTypeTree", num_leaves); 
    return 0 ; 
}

int sn_test::CommonOperatorTypeTree()
{
    if(sn::level() > 0) std::cout << "[ sn_test::CommonOperatorTypeTree " << std::endl ; 
    int N = 32 ; 
    for(int nl=1 ; nl < N ; nl++) sn_test::CommonOperatorTypeTree(nl); 
    if(sn::level() > 0) std::cout << "] sn_test::CommonOperatorTypeTree " << std::endl ; 
    check_LEAK("CommonOperatorTypeTree"); 
    return 0 ; 
}


int sn_test::label()
{
    int it = 3 ; 
    if(sn::level() > 0) std::cout << "[ sn_test::label it " << it  << std::endl ; 

    sn* t = manual_tree(it); 

    t->labeltree(); 

    if(sn::level() > 1) std::cout << t->render(3) ; 

    if(!sn::LEAK) delete t ; 
    if(sn::level() > 0) std::cout << "] sn_test::label it " << it  << std::endl ; 
    check_LEAK("label"); 
    return 0 ; 
}

int sn_test::positivize()
{
    int it = ssys::getenvint("TREE", 3) ; 
    if(sn::level() > 0) std::cout << "[ sn_test::positivize it " << it  << std::endl ; 

    sn* t = manual_tree(it); 

    int mode = ssys::getenvint("MODE", 4) ; 

    t->labeltree(); 
    if(sn::level() > 1) std::cout << t->render(mode) ; 

    t->positivize(); 
    if(sn::level() > 1) std::cout << t->render(mode) ; 

    if(!sn::LEAK) delete t ; 
    if(sn::level() > 0) std::cout << "] sn_test::positivize it " << it  << std::endl ; 
    check_LEAK("positivize"); 
    return 0 ; 
}

int sn_test::pool()
{
    if(sn::level() > 0) std::cout << "[ sn_test::pool " << std::endl ; 
    assert( sn::pool->size() == 0  );  
    sn* a = sn::Zero() ; 


    assert( sn::pool->size() == 1  );  
    sn* b = sn::Zero() ; 

    assert( sn::pool->size() == 2  );  
    sn* c = sn::Zero() ; 

    assert( sn::pool->size() == 3  );  


    if(sn::level() > 1) std::cout << sn::Desc() ; 


    delete c ; 
    assert( sn::pool->size() == 2  );  

    delete a ; 
    assert( sn::pool->size() == 1  );  

    delete b ; 
    assert( sn::pool->size() == 0  );  

    if(sn::level() > 0) std::cout << "] sn_test::pool " << std::endl ; 
    check_LEAK("pool"); 
    return 0 ; 
}

int sn_test::Simple()
{
    int it = 3 ; 
    if(sn::level() > 0) std::cout << "[ sn_test::Simple it " << it << std::endl ; 

    sn* t = manual_tree(it); 

    t->labeltree(); 

    if(sn::level() > 1) std::cout << t->render(5) ; 
    if(sn::level() > 1) std::cout << sn::Desc() ; 

    if(!sn::LEAK) delete t ; 
    if(sn::level() > 0) std::cout << "] sn_test::Simple it " << it << std::endl ; 
    check_LEAK("Simple"); 
    return 0 ; 
}

int sn_test::set_left()
{
    if(sn::level() > 0) std::cout << "[ sn_test::set_left" << std::endl ; 

    sn* l = sn::Prim(CSG_SPHERE); 
    sn* r = sn::Prim(CSG_BOX3); 
    sn* b = sn::Create(CSG_UNION, l, r ); 

    sn* al = sn::Prim(CSG_BOX3) ; 

    b->set_left(al, false); 

    if(sn::level() > 1) std::cout << sn::Desc() ; 

    delete b ; 
    if(sn::level() > 1) std::cout << sn::Desc() ; 

    if(sn::level() > 0) std::cout << "] sn_test::set_left" << std::endl ; 
    check_LEAK("set_left"); 
    return 0 ; 
}

int sn_test::serialize_0()
{
    int lev = 0 ; 
    int it = 3 ; 
    if(sn::level() > lev) std::cout << "[ sn_test::serialize_0 it " << it  << std::endl ; 

    sn* t = manual_tree(it); 
    if(sn::level() > lev) std::cout << t->render(5) ; 
    if(sn::level() > lev) std::cout << sn::Desc(); 

    int num_root = sn::pool->num_root() ; 
    if(sn::level() > lev) std::cout << " num_root " << num_root << std::endl ; 
 
    std::vector<_sn> buf ; 
    sn::pool->serialize_(buf); 

    delete t ; 

    NP* a = NPX::ArrayFromVec_<int, _sn>(buf, _sn::ITEM) ; 

    //NP* a = NP::Make<int>( buf.size(), _sn::NV ) ; 
    //a->read2<int>((int*)buf.data()); 

    if(sn::level() > lev) std::cout << " save to " << FOLD << "/" << sn::NAME << std::endl ; 
    a->save(FOLD, sn::NAME); 

    if(sn::level() > lev) std::cout << "] sn_test::serialize buf.size() " << buf.size()  << std::endl ; 
    if(sn::level() > lev) std::cout << sn::pool->Desc(buf) << std::endl ; 

    check_LEAK("serialize_0"); 
    return 0 ; 
}

int sn_test::serialize_1()
{
    int lev = 0 ; 
    int it = 3 ; 
    if(sn::level() > lev) std::cout << "[ sn_test::serialize_1 it " << it  << std::endl ; 

    sn* t = manual_tree(it); 
    if(sn::level() > lev) std::cout << t->render(5) ; 
    if(sn::level() > lev) std::cout << sn::Desc(); 

    int num_root = sn::pool->num_root() ; 
    if(sn::level() > lev) std::cout << " num_root " << num_root << std::endl ; 

    NP* a = sn::pool->serialize<int>() ; 

    delete t ; 

    if(sn::level() > lev) std::cout << " save to " << FOLD << "/" << sn::NAME << std::endl ; 

    a->save(FOLD, sn::NAME); 

    check_LEAK("serialize_1"); 
    return 0 ; 
}

int sn_test::import_0()
{
    int lev = 0 ; 
    if(sn::level() > lev) std::cout << "[ sn_test::import_0 " << std::endl ; 

    if(sn::level() > lev) std::cout << " load from " << FOLD << "/" << sn::NAME << std::endl ; 
    NP* a = NP::Load(FOLD, sn::NAME );
    std::vector<_sn> buf(a->shape[0]) ; 
    a->write<int>((int*)buf.data()); 



    sn::pool->import_(buf); 

    if(sn::level() > lev) std::cout << sn::pool->Desc(buf) << std::endl ; 

    int num_root = sn::pool->num_root() ; 
    if(sn::level() > lev) std::cout << " num_root " << num_root << std::endl ; 
    if(sn::level() > lev) std::cout << sn::Desc(); 


    sn* root = sn::pool->get_root(0) ; 
    if(root && sn::level() > lev) std::cout << root->render(5); 
    delete root ; 

    if(sn::level() > lev) std::cout << "] sn_test::import_0 " << std::endl ; 
    check_LEAK("import_0"); 
    return 0 ; 
}

int sn_test::import_1()
{
    int lev = 0 ; 
    if(sn::level() > lev) std::cout << "[ sn_test::import_1 " << std::endl ; 

    if(sn::level() > lev) std::cout << " load from " << FOLD << "/" << sn::NAME << std::endl ; 
    NP* a = NP::Load(FOLD, sn::NAME );

    sn::pool->import<int>(a); 

    int num_root = sn::pool->num_root() ; 
    if(sn::level() > lev) std::cout << " num_root " << num_root << std::endl ; 
    if(sn::level() > lev) std::cout << sn::Desc(); 

    sn* root = sn::pool->get_root(0) ; 
    if(root && sn::level() > lev) std::cout << root->render(5); 
    delete root ; 

    if(sn::level() > lev) std::cout << "] sn_test::import_1 " << std::endl ; 
    check_LEAK("import_1"); 
    return 0 ; 
}







int sn_test::dtor_0()
{
    sn* n = sn::Zero(); 
    delete n ; 
    check_LEAK("dtor_0"); 
    return 0 ; 
}
int sn_test::dtor_1()
{
    sn* a = sn::Prim(100); 
    sn* b = sn::Prim(101);
    sn* c = sn::Create(1, a, b ); 
    delete c ; 
    check_LEAK("dtor_1"); 
    return 0 ; 
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


int sn_test::set_child()
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

    check_LEAK("set_child"); 
    return 0 ; 
}

int sn_test::deepcopy_0()
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

    check_LEAK("deepcopy_0"); 
    return 0 ; 
}

/**
sn_test::deepcopy_1_leaking
-----------------------------

NB the child vector is shallow copied by this default copy ctor
which causes the cp and c to both think they own a and b 
which will cause ownership isses when delete 

hence cannot easily clean up this situation

**/

int sn_test::deepcopy_1_leaking()
{
    sn* a = sn::Prim(100); 
    sn* b = sn::Prim(101); 
    sn* c = sn::Create(1, a, b ); 

    std::cout << "c\n " << c->desc_child() << std::endl ; 

    sn* cp = new sn(*c) ; 
    std::cout << "cp\n" << cp->desc_child() << std::endl ; 

    //check_LEAK("deepcopy_1_leaking"); 
    return 0 ; 
}

int sn_test::next_sibling()
{
    int lev = 0 ; 

    sn* a = sn::Prim(100); 
    sn* b = sn::Prim(101); 
    sn* c = sn::Create(1, a, b ); 

    int ia = a->sibling_index() ;  
    int ib = b->sibling_index() ;  

    if(sn::level() > lev) std::cerr << "sn_test::next_sibling ia  " << ia  << std::endl ;     
    if(sn::level() > lev) std::cerr << "sn_test::next_sibling ib  " << ib  << std::endl ;     
    
    const sn* x = a->next_sibling() ; 
    const sn* y = b->next_sibling() ; 
    const sn* z = c->next_sibling() ; 

    if(sn::level() > lev) std::cerr << "sn_test::next_sibling x: " << ( x ? "Y" : "N" ) << std::endl ;     
    if(sn::level() > lev) std::cerr << "sn_test::next_sibling y: " << ( y ? "Y" : "N" ) << std::endl ;     
    if(sn::level() > lev) std::cerr << "sn_test::next_sibling z: " << ( z ? "Y" : "N" ) << std::endl ;     

    assert( x == b ); 
    assert( y == nullptr ); 
    assert( z == nullptr ); 

    delete c ; 

    check_LEAK("next_sibling"); 
    return 0 ; 
}

int sn_test::Serialize()
{
    int lev = -1 ; 
    int it = 4 ; 
    if(sn::level() > lev) std::cout << "[ sn_test::Serialize it " << it  << std::endl ; 

    sn* t = manual_tree(it); 

    if(sn::level() > lev) std::cout << t->render(5) ; 
    if(sn::level() > lev) std::cout << sn::Desc(); 

    NPFold* fold = s_csg::Serialize() ; 

    delete t ; 

    fold->save(FOLD) ; 
    
    if(sn::level() > lev) std::cout << "] sn_test::Serialize it " << it  << std::endl ; 

    check_LEAK("Serialize"); 

    return 0 ; 
}


int sn_test::Import()
{
    NPFold* fold = NPFold::Load(FOLD) ; 
 
    s_csg::Import( fold ); 


    std::cout << s_tv::pool->desc() ;  

    sn* t = sn::pool->get_root(0) ; 

    std::cout << t->render(0) ; 

    sn* r = t->last_child(); 

    s_tv* xform = r->xform ; 

    std::cout << ( xform ? xform->desc() : "xform-null" ) << std::endl ; 

    delete t ; 

    check_LEAK("Import"); 

    return 0 ; 
}



int sn_test::OrderPrim_()
{
    std::vector<sn*> prim0 ; 
    sn* a = sn::Cylinder(100., -200., -100. );      
    sn* b = sn::Cylinder( 50., -100.,    0. );      
    sn* c = sn::Cylinder( 60.,    0.,  100. );      
    sn* d = sn::Cylinder( 70.,  100.,  200. );      
   
    prim0.push_back(a) ; 
    prim0.push_back(b) ; 
    prim0.push_back(c) ; 
    prim0.push_back(d) ; 

    std::reverse( prim0.begin(), prim0.end() ); 

    sn* root = sn::UnionTree(prim0); 
    root->setAABB_TreeFrame_All(); 

    bool reverse = false ; 

    std::cout 
        << "sn_test::OrderPrim"
        << std::endl
        << " root->desc_prim_all() "
        << root->desc_prim_all(reverse) 
        << std::endl
        ;

    std::vector<const sn*> prim ; 
    root->collect_prim(prim); 
    assert(prim.size() == prim0.size() ); 

    std::cout 
        << " sn::DescPrim(prim) asis from  : root->collect_prim(prim)  "
        << std::endl
        <<  sn::DescPrim(prim)
        << std::endl
        ;

    bool ascending = true ; 

    sn::OrderPrim<const sn>( prim, sn::AABB_ZMin, ascending  ); 
    std::cout 
        << " sn::DescPrim(prim) after : sn::OrderPrim<const sn>( prim, sn::AABB_ZMin ) "
        << std::endl
        <<  sn::DescPrim(prim)
        << std::endl
        ;

    sn::OrderPrim<const sn>( prim, sn::AABB_XMin, ascending  ); 
    std::cout 
        << " sn::DescPrim(prim) after : sn::OrderPrim<const sn>( prim, sn::AABB_XMin ) "
        << std::endl
        <<  sn::DescPrim(prim)
        << std::endl
        ;

    sn::OrderPrim<const sn>( prim, sn::AABB_XMax, ascending  ); 
    std::cout 
        << " sn::DescPrim(prim) after : sn::OrderPrim<const sn>( prim, sn::AABB_XMax ) "
        << std::endl
        <<  sn::DescPrim(prim)
        << std::endl
        ;

    return 0 ; 
}


int sn_test::ALL()
{
    int rc = 0 ; 
    rc += BinaryTreeHeight();
    rc += ZeroTree();
    rc += CommonOperatorTypeTree();
    rc += Serialize();
    rc += label();
    rc += positivize();
    rc += pool();
    rc += Simple();
    rc += set_left();
    rc += serialize_0();
    rc += serialize_1();
    rc += import_0();
    rc += import_1();
    rc += dtor_0();
    rc += dtor_1();
    rc += set_child();
    rc += deepcopy_0();
    //rc += deepcopy_1_leaking();
    rc += next_sibling();
    rc += Serialize();

    return rc ; 
}


/**
sn_test::main
------------------

Before fixing sn::deepcopy with sn::disown_child some of the below 
CommonOperatorTypeTree tests segmented or hung
**/

int sn_test::main()
{
    const char* TEST = ssys::getenvvar("TEST","ALL") ; 

    int rc = 0 ; 
    if(      strcmp(TEST, "BinaryTreeHeight") == 0 ) rc = BinaryTreeHeight() ;
    else if( strcmp(TEST, "ZeroTree")==0) rc = ZeroTree();
    else if( strcmp(TEST, "CommonOperatorTypeTree")==0) rc = CommonOperatorTypeTree();
    else if( strcmp(TEST, "CommonOperatorTypeTree1")==0) rc = CommonOperatorTypeTree(1);
    else if( strcmp(TEST, "CommonOperatorTypeTree4")==0) rc = CommonOperatorTypeTree(4);
    else if( strcmp(TEST, "CommonOperatorTypeTree32")==0) rc = CommonOperatorTypeTree(32);
    else if( strcmp(TEST, "Serialize")==0)  rc = Serialize(); 
    else if( strcmp(TEST, "label")==0)      rc = label(); 
    else if( strcmp(TEST, "positivize")==0) rc = positivize(); 
    else if( strcmp(TEST, "pool")==0)       rc = pool(); 
    else if( strcmp(TEST, "Simple")==0)     rc = Simple(); 
    else if( strcmp(TEST, "set_left")==0)     rc = set_left(); 
    //else if( strcmp(TEST, "serialize_0")==0)     rc = serialize_0(); 
    else if( strcmp(TEST, "serialize_1")==0)     rc = serialize_1(); 
    //else if( strcmp(TEST, "import_0")==0)     rc = import_0(); 
    else if( strcmp(TEST, "import_1")==0)     rc = import_1(); 
    else if( strcmp(TEST, "dtor_0")==0)     rc = dtor_0(); 
    else if( strcmp(TEST, "dtor_1")==0)     rc = dtor_1(); 
    else if( strcmp(TEST, "set_child")==0)     rc = set_child(); 
    else if( strcmp(TEST, "deepcopy_0")==0)     rc = deepcopy_0(); 
    else if( strcmp(TEST, "deepcopy_1_leaking")==0)  rc = deepcopy_1_leaking(); 
    else if( strcmp(TEST, "next_sibling")==0)        rc = next_sibling(); 
    else if( strcmp(TEST, "Serialize")==0)        rc = Serialize(); 
    else if( strcmp(TEST, "Import")==0)        rc = Import(); 
    else if( strcmp(TEST, "OrderPrim")==0)     rc = OrderPrim_(); 
    else if( strcmp(TEST, "ALL")==0)           rc = ALL(); 

    return rc ;
}

int main(int argc, char** argv)
{
    s_csg* _csg = new s_csg ; 
    std::cout << _csg->brief() ; 
    return sn_test::main() ; 
}

