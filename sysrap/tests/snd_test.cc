// ./snd_test.sh

#include <iostream>

#include "stree.h"
#include "snd.hh"
#include "scsg.hh"
#include "NPFold.h"


const char* FOLD = getenv("FOLD"); 

void test_empty()
{
    std::cout << "test_empty" << std::endl ;     
    snd a = {} ; 
    a.setTC(0); 
    a.setPA( 1., 2., 3., 4., 5., 6. ); 
    snd b = snd::Sphere(100.) ; 
    snd c = snd::ZSphere(100., -10.,  10. ) ; 

    // NB setting param and aabb adds to the pools 
    //    even when no nodes are added 

    std::cout << " a " << a.desc() << std::endl ; 
    std::cout << " b " << b.desc() << std::endl ; 
    std::cout << " c " << c.desc() << std::endl ; 
}
void test_Add()
{
    std::cout << "test_Add" << std::endl ;     
    snd a = {} ; 
    a.setTC(0); 
    a.setPA( 1., 2., 3., 4., 5., 6. ); 
    snd b = snd::Sphere(100.) ; 
    snd c = snd::ZSphere(100., -10.,  10. ) ; 

    snd::Add(a); 
    snd::Add(b); 
    snd::Add(c); 
}

int main(int argc, char** argv)
{
    stree st ; 

    if(argc == 1) 
    {
        //test_empty(); 
        test_Add(); 

        NPFold* fold = snd::Serialize() ; 
        std::cout << " save snd to FOLD " << FOLD << std::endl ;  
        fold->save(FOLD); 
    }
    else
    {
        NPFold* fold = NPFold::Load(FOLD) ; 
        std::cout << " load snd from FOLD " << FOLD << std::endl ;  
        snd::Import(fold);  
    }
        
    std::cout << snd::Desc() ; 
    return 0 ; 
}

