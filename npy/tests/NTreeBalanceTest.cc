// TEST=NTreeBalanceTest om-t

#include <cstdlib>
#include <cassert>
#include <vector>

#include "OPTICKS_LOG.hh"

#include "NBox.hpp"
#include "NSphere.hpp"
#include "NNode.hpp"
#include "NTreeAnalyse.hpp"
#include "NTreePositive.hpp"
#include "NTreeBalance.hpp"
#include "NTreeProcess.hpp"


nnode* make_tree_0()
{
    nnode* a = make_sphere(0,0,-50,100) ;  
    nnode* b = make_sphere(0,0, 50,100) ;  
    nnode* c = make_box(0,0, 50,100) ;  
    nnode* d = make_box(0,0,  0,100) ;  
    nnode* e = make_box(0,0,  0,100) ;  

    nnode* ab = nnode::make_operator( CSG_UNION, a, b );
    nnode* de = nnode::make_operator( CSG_DIFFERENCE, d, e );
    nnode* cde = nnode::make_operator( CSG_DIFFERENCE, c, de );
    nnode* abcde = nnode::make_operator( CSG_INTERSECTION, ab, cde );

    return abcde ; 
}


nnode* make_tree_1()
{
    nnode* a = make_sphere(0,0,-50,100) ;  
    nnode* b = make_sphere(0,0,-50,100) ;  
    nnode* c = make_sphere(0,0,-50,100) ;  
    nnode* d = make_sphere(0,0,-50,100) ;  
    nnode* e = make_sphere(0,0,-50,100) ;  
    nnode* f = make_sphere(0,0,-50,100) ;  
    nnode* g = make_sphere(0,0,-50,100) ;  

    nnode* ab = nnode::make_operator( CSG_DIFFERENCE, a, b );
    nnode* abc = nnode::make_operator( CSG_DIFFERENCE, ab, c );
    nnode* abcd = nnode::make_operator( CSG_DIFFERENCE, abc, d );
    nnode* abcde = nnode::make_operator( CSG_DIFFERENCE, abcd, e );
    nnode* abcdef = nnode::make_operator( CSG_DIFFERENCE, abcde, f );
    nnode* abcdefg = nnode::make_operator( CSG_DIFFERENCE, abcdef, g );

    return abcdefg ; 
}

/**
make_tree_2
--------------

                                                                                      un            

                                                                              un              di    

                                                                      un          cy      cy      cy

                                                              un          cy                        

                                                      un          cy                                

                                              un          cy                                        

                                      un          cy                                                

                              un          cy                                                        

                      un          cy                                                                

              un          cy                                                                        

      di          cy                                                                                

  cy      cy                                    


**/


nnode* make_tree_2()
{
    nnode* a = make_sphere(0,0,-50,100) ;  
    nnode* b = make_sphere(0,0,-50,100) ;  
    nnode* c = make_sphere(0,0,-50,100) ;  
    nnode* d = make_sphere(0,0,-50,100) ;  
    nnode* e = make_sphere(0,0,-50,100) ;  
    nnode* f = make_sphere(0,0,-50,100) ;  
    nnode* g = make_sphere(0,0,-50,100) ;  
    nnode* h = make_sphere(0,0,-50,100) ;  
    nnode* i = make_sphere(0,0,-50,100) ;  
    nnode* j = make_sphere(0,0,-50,100) ;  
    nnode* k = make_sphere(0,0,-50,100) ;  
    nnode* l = make_sphere(0,0,-50,100) ;  
    nnode* m = make_sphere(0,0,-50,100) ;  

    nnode* ab = nnode::make_operator( CSG_DIFFERENCE, a, b );

    nnode* abc = nnode::make_operator( CSG_UNION, ab, c );
    nnode* abcd = nnode::make_operator( CSG_UNION, abc, d );
    nnode* abcde = nnode::make_operator( CSG_UNION, abcd, e );
    nnode* abcdef = nnode::make_operator( CSG_UNION, abcde, f );
    nnode* abcdefg = nnode::make_operator( CSG_UNION, abcdef, g );
    nnode* abcdefgh = nnode::make_operator( CSG_UNION, abcdefg, h );
    nnode* abcdefghi = nnode::make_operator( CSG_UNION, abcdefgh, i );
    nnode* abcdefghij = nnode::make_operator( CSG_UNION, abcdefghi, j );
    nnode* abcdefghijk = nnode::make_operator( CSG_UNION, abcdefghij, k );

    nnode* lm = nnode::make_operator( CSG_DIFFERENCE, l, m );

    nnode* abcdefghijklm = nnode::make_operator( CSG_UNION, abcdefghijk, lm );
    nnode* root = abcdefghijklm  ; 

    return root ; 
}

nnode* make_tree( int i )
{
    nnode* root = NULL ; 
    switch(i)
    {
        case 0: root = make_tree_0() ; break ; 
        case 1: root = make_tree_1() ; break ; 
        case 2: root = make_tree_2() ; break ; 
    }
    assert( root ) ; 
    return root ; 
}




void test_balance_0(nnode* tree)
{
    NTreeBalance<nnode> bal(tree);  // writes depth, subdepth to all nodes

    LOG(info) << tree->desc() ; 
    LOG(info) << NTreeAnalyse<nnode>::Desc(tree) ;
    LOG(info) << " ops: " << bal.operatorsDesc(0) ; 
    assert( bal.is_positive_form() == false );


    NTreePositive<nnode> pos(tree) ; 

    LOG(info) << NTreeAnalyse<nnode>::Desc(tree) ; 
    LOG(info) << " ops: " << bal.operatorsDesc(0) ; 
    assert( bal.is_positive_form() == true );


    std::vector<nnode*> subs ; 
    std::vector<nnode*> otherprim ; 
    unsigned subdepth = 0 ; 
    bal.subtrees( subs, subdepth, otherprim );  // 
    LOG(info) 
        << " subdepth " << subdepth 
        << " subs " << subs.size() 
        ; 

    for(unsigned i=0 ; i < subs.size() ; i++)
    {
        nnode* sub = subs[i] ;
        LOG(info) << NTreeAnalyse<nnode>::Desc(sub) ;
    }
}


void test_balance(nnode* tree)
{
    LOG(info) << "tree initial \n" << NTreeAnalyse<nnode>::Desc(tree) ; 

    NTreePositive<nnode> pos(tree) ; // eliminate CSG_DIFFERENCE by DeMorgan rules, some leaves may be complemented 
    LOG(info) << "tree positivized\n" << NTreeAnalyse<nnode>::Desc(tree) ; 

    NTreeBalance<nnode> bal(tree);   // writes depth, subdepth to all nodes

    nnode* balanced = bal.create_balanced(); 

    LOG(info) << "tree balanced\n" << NTreeAnalyse<nnode>::Desc(balanced) ; 
}

void test_process(nnode* tree)
{
    LOG(info) << "tree initial \n" << NTreeAnalyse<nnode>::Desc(tree) ; 

    NTreeProcess<nnode> proc(tree); 

    nnode* result = proc.result ; 
    LOG(info) << "tree result \n" << NTreeAnalyse<nnode>::Desc(result) ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    int it = 2 ;  
    nnode* tree = make_tree(it) ; 

    //test_balance_0(tree);
    //test_balance(tree);
    test_process(tree);

    return 0 ; 
}


