#include <cstdlib>
#include <cassert>
#include <vector>

#include "OPTICKS_LOG.hh"

#include "NBox.hpp"
#include "NSphere.hpp"
#include "NNode.hpp"
#include "NNodeAnalyse.hpp"
#include "NTreePositive.hpp"
#include "NTreeBalance.hpp"
#include "NTreeProcess.hpp"


nnode* make_tree_0()
{
    nnode* a = new nsphere(make_sphere(0,0,-50,100)) ;  
    nnode* b = new nsphere(make_sphere(0,0, 50,100)) ;  
    nnode* c = new nbox(make_box(0,0, 50,100)) ;  
    nnode* d = new nbox(make_box(0,0,  0,100)) ;  
    nnode* e = new nbox(make_box(0,0,  0,100)) ;  

    nnode* ab = nnode::make_operator_ptr( CSG_UNION, a, b );
    nnode* de = nnode::make_operator_ptr( CSG_DIFFERENCE, d, e );
    nnode* cde = nnode::make_operator_ptr( CSG_DIFFERENCE, c, de );
    nnode* abcde = nnode::make_operator_ptr( CSG_INTERSECTION, ab, cde );

    return abcde ; 
}


nnode* make_tree()
{
    nnode* a = new nsphere(make_sphere(0,0,-50,100)) ;  
    nnode* b = new nsphere(make_sphere(0,0,-50,100)) ;  
    nnode* c = new nsphere(make_sphere(0,0,-50,100)) ;  
    nnode* d = new nsphere(make_sphere(0,0,-50,100)) ;  
    nnode* e = new nsphere(make_sphere(0,0,-50,100)) ;  
    nnode* f = new nsphere(make_sphere(0,0,-50,100)) ;  
    nnode* g = new nsphere(make_sphere(0,0,-50,100)) ;  

    nnode* ab = nnode::make_operator_ptr( CSG_DIFFERENCE, a, b );
    nnode* abc = nnode::make_operator_ptr( CSG_DIFFERENCE, ab, c );
    nnode* abcd = nnode::make_operator_ptr( CSG_DIFFERENCE, abc, d );
    nnode* abcde = nnode::make_operator_ptr( CSG_DIFFERENCE, abcd, e );
    nnode* abcdef = nnode::make_operator_ptr( CSG_DIFFERENCE, abcde, f );
    nnode* abcdefg = nnode::make_operator_ptr( CSG_DIFFERENCE, abcdef, g );

    return abcdefg ; 
}


void test_balance_0()
{
    nnode* tree = make_tree() ; 
    NTreeBalance<nnode> bal(tree);  // writes depth, subdepth to all nodes

    LOG(info) << tree->desc() ; 
    LOG(info) << NNodeAnalyse<nnode>::Desc(tree) ;
    LOG(info) << " ops: " << bal.operatorsDesc(0) ; 
    assert( bal.is_positive_form() == false );


    NTreePositive<nnode> pos(tree) ; 

    LOG(info) << NNodeAnalyse<nnode>::Desc(tree) ; 
    LOG(info) << " ops: " << bal.operatorsDesc(0) ; 
    assert( bal.is_positive_form() == true );


    std::vector<nnode*> subs ; 
    unsigned subdepth = 0 ; 
    bal.subtrees( subs, subdepth );  // 
    LOG(info) 
        << " subdepth " << subdepth 
        << " subs " << subs.size() 
        ; 

    for(unsigned i=0 ; i < subs.size() ; i++)
    {
        nnode* sub = subs[i] ;
        LOG(info) << NNodeAnalyse<nnode>::Desc(sub) ;
    }
}


void test_balance()
{
    nnode* tree = make_tree() ; 
    LOG(info) << "tree initial \n" << NNodeAnalyse<nnode>::Desc(tree) ; 

    NTreePositive<nnode> pos(tree) ; // eliminate CSG_DIFFERENCE by DeMorgan rules, some leaves may be complemented 
    LOG(info) << "tree positivized\n" << NNodeAnalyse<nnode>::Desc(tree) ; 

    NTreeBalance<nnode> bal(tree);   // writes depth, subdepth to all nodes

    nnode* balanced = bal.create_balanced(); 

    LOG(info) << "tree balanced\n" << NNodeAnalyse<nnode>::Desc(balanced) ; 
}

void test_process()
{
    nnode* tree = make_tree() ; 
    LOG(info) << "tree initial \n" << NNodeAnalyse<nnode>::Desc(tree) ; 

    NTreeProcess<nnode> proc(tree); 

    nnode* result = proc.result ; 
    LOG(info) << "tree result \n" << NNodeAnalyse<nnode>::Desc(result) ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //test_balance_0();
    //test_balance();
    test_process();

    return 0 ; 
}


