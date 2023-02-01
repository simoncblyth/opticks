// ./snd_test.sh

#include <iostream>

#include "stree.h"
#include "snd.hh"
#include "scsg.hh"
#include "NPFold.h"
#include "OpticksCSG.h"

const char* TEST = getenv("TEST"); 
const char* FOLD = getenv("FOLD"); 

void do_Add()
{
    std::cout << "test_Add" << std::endl ;     
    int a = snd::Zero( 1., 2., 3., 4., 5., 6. ); 
    int b = snd::Sphere(100.) ; 
    int c = snd::ZSphere(100., -10.,  10. ) ; 

    std::vector<int> prims = {a,b,c} ; 
    int d = snd::Compound(CSG_CONTIGUOUS, prims ) ; 

    std::cout << "test_Add :  Desc dumping " << std::endl; 

    std::cout << " a " << snd::Desc(a) << std::endl ; 
    std::cout << " b " << snd::Desc(b) << std::endl ; 
    std::cout << " c " << snd::Desc(c) << std::endl ; 
    std::cout << " d " << snd::Desc(d) << std::endl ; 
}

void test_save()
{
    do_Add(); 

    std::cout << snd::Desc() ; 
    NPFold* fold = snd::Serialize() ; 
    std::cout << " save snd to FOLD " << FOLD << std::endl ;  
    fold->save(FOLD); 
}

void test_load()
{
    NPFold* fold = NPFold::Load(FOLD) ; 
    std::cout << " load snd from FOLD " << FOLD << std::endl ;  
    snd::Import(fold);  
    std::cout << snd::Desc() ; 
}


void test_max_depth_(int n, int xdepth)
{
    const snd* nd = snd::GetNode(n);
    assert( nd );  
    assert( nd->max_depth() == xdepth );  
    assert( snd::GetMaxDepth(n) == xdepth );  

    std::cout << "test_max_depth_ n " << n << " xdepth " << xdepth << std::endl ;  
}

void test_max_depth()
{
    std::cout << "test_max_depth" << std::endl ;     

    int a = snd::Sphere(100.) ; 
    int b = snd::Sphere(100.) ; 
    int c = snd::Boolean(CSG_UNION, a, b ) ; 
    int d = snd::Box3(100.) ; 
    int e = snd::Boolean(CSG_UNION, c, d ) ; 
    int f = snd::Box3(100.) ; 
    int g = snd::Boolean(CSG_UNION, f, e ) ; 

    // NB setLVID never called here : so no snd has been "declared" as root 

    test_max_depth_( a, 0 ); 
    test_max_depth_( b, 0 ); 
    test_max_depth_( c, 1 ); 
    test_max_depth_( d, 0 );
    test_max_depth_( e, 2 );
    test_max_depth_( f, 0 );
    test_max_depth_( g, 3 );
} 


void test_num_node_(int n, int xnn)
{
    const snd* nd = snd::GetNode(n);

    int nnn = nd->num_node() ; 
    int gnn = snd::GetNumNode(n) ; 

    std::cout 
        << "test_num_node_ n " << n 
        << " xnn " << xnn 
        << " nnn " << nnn 
        << " gnn " << gnn 
        << std::endl
        ;  

    assert( nd );  
    assert( nnn == xnn );  
    assert( gnn == xnn );  

}

/**
                   g 
            e            f
        c       d
     a     b

**/

void test_num_node()
{
    std::cout << "test_num_node" << std::endl ;     

    int a = snd::Sphere(100.) ; 
    int b = snd::Sphere(100.) ; 
    int c = snd::Boolean(CSG_UNION, a, b ) ; 
    int d = snd::Box3(100.) ; 
    int e = snd::Boolean(CSG_UNION, c, d ) ; 
    int f = snd::Box3(100.) ; 
    int g = snd::Boolean(CSG_UNION, f, e ) ; 

    // NB setLVID never called here : so no snd has been "declared" as root 

    test_num_node_( a, 1 ); 
    test_num_node_( b, 1 ); 
    test_num_node_( c, 3 ); 
    test_num_node_( d, 1 );
    test_num_node_( e, 5 );
    test_num_node_( f, 1 );
    test_num_node_( g, 7 );
} 




int main(int argc, char** argv)
{
    stree st ; 

    if(     strcmp(TEST, "save")==0)      test_save(); 
    else if(strcmp(TEST, "load")==0)      test_load(); 
    else if(strcmp(TEST, "max_depth")==0) test_max_depth() ; 
    else if(strcmp(TEST, "num_node")==0)  test_num_node() ; 
    else std::cerr << " TEST not matched " << TEST << std::endl ; 
        
    return 0 ; 
}

