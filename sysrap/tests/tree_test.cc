// ./tree_test.sh 

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <map>

#include "tree.h"

void test_flip( nd* n )
{
    n->value = -n->value ;  // easy way to flip the nodes value 
}

void test_placement_new( nd* n )
{
    nd* p = new (n) nd( -n->value, n->left, n->right) ; 
    assert( p == n ); 
    // placement new creation of new object to replace the nd at same location 
}


void test_cuts(int height0)
{
    int count0 = tree::NumNode(height0) ; 

    for( int cut=count0 ; cut > 0 ; cut-- )
    {
        tree* t = tree::make_complete(height0) ;
        t->initvalue(PRE); 

        int count0 = t->num_node() ; 
        t->apply_cut(cut); 
        printf("count0 %d cut %d t.desc %s\n", count0, cut, t->desc() ); 
        t->draw(); 
    }
}

void test_unbalanced(int numprim0, int order, const char* msg=nullptr)
{
    tree* t = tree::make_unbalanced(numprim0) ;
    t->initvalue(order); 
    t->draw(msg); 
}
void test_unbalanced(int numprim)
{
    test_unbalanced(numprim, POST); 
    test_unbalanced(numprim, RPRE,  "RPRE is opposite order to POST" ); 

    test_unbalanced(numprim, PRE); 
    test_unbalanced(numprim, RPOST, "RPOST is opposite order to PRE"); 

    test_unbalanced(numprim, IN); 
    test_unbalanced(numprim, RIN,   "RIN is opposite order to IN"); 
}

void test_complete(int numprim0, int order, const char* msg=nullptr)
{
    tree* t = tree::make_complete(numprim0) ; 
    t->initvalue(order); 
    t->draw(msg); 
}
void test_complete(int numprim)
{
    test_complete(numprim, POST); 
    test_complete(numprim, RPRE,  "RPRE is opposite order to POST" ); 

    test_complete(numprim, PRE); 
    test_complete(numprim, RPOST, "RPOST is opposite order to PRE"); 

    test_complete(numprim, IN); 
    test_complete(numprim, RIN,   "RIN is opposite order to IN"); 
}


void test_cuts_unbalanced(int numprim0)
{
    for(int i=0 ; i < 20 ; i++)
    {
        tree* t = tree::make_unbalanced(numprim0) ; 
        t->initvalue(POST); 
        //t->draw(); 

        int count0 = t->num_node() ; 
        int cut = count0 - i ;  
        if( cut < 1 ) break ; 
        t->apply_cut(cut); 
        printf("count0 %d cut %d t.desc %s\n", count0, cut, t->desc() ); 

        t->draw(); 
    }
}

void test_cut_unbalanced(int numprim0, int cut)
{
    printf("test_cut_unbalanced numprim0 %d cut %d \n", numprim0, cut ); 

    tree* t = tree::make_unbalanced(numprim0) ; 
    t->initvalue(POST); 
    t->draw("before cut"); 
    t->apply_cut(cut); 
    t->draw("after cut"); 
}

void test_no_remaining_nodes()
{
    int height0 = 3 ; 
    tree* t = tree::make_complete(height0) ;
    t->initvalue(PRE); 
    t->draw(); 
    t->verbose = true ; 

    printf("t.desc %s \n", t->desc() );  

    int count0 = t->num_node() ; 
    int cut = 3 ; 
    t->apply_cut(cut); 
    printf("count0 %d cut %d t.desc %s\n", count0, cut, t->desc() ); 
    t->draw(); 
}

void test_unbalanced_cut()
{
    tree* t = tree::make_unbalanced(8) ;
    t->initvalue(POST); 
    t->draw("before cut"); 

    t->apply_cut(8); 
    t->draw("after cut"); 
}

void test_complete_initvalue()
{
    tree* t = tree::make_complete(4) ;
    t->initvalue(POST); 
    t->draw(); 
}



int main(int argc, char**argv )
{
    /*
    test_unbalanced(8); 
    test_complete(4); 

    test_cuts(4); 
    test_cuts(3); 
    test_no_remaining_nodes();

    test_cut_unbalanced(8, 8); 
    test_cuts_unbalanced(8); 
    
    test_unbalanced_cut(); 
    test_complete_initvalue(); 
    */

    test_unbalanced(8); 


    return 0 ; 
}
