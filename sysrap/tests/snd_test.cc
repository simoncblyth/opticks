// ./snd_test.sh

#include <iostream>

#include "stree.h"
#include "snd.hh"
#include "scsg.hh"
#include "NPFold.h"
#include "OpticksCSG.h"

const char* FOLD = getenv("FOLD"); 

void test_Add()
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

int main(int argc, char** argv)
{
    stree st ; 

    if(argc == 1) 
    {
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

