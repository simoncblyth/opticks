/**
stree_loadsave_test.cc
=========================


**/


#include "stree.h"

int main(int argc, char** argv)
{
    const char* ss = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim" ; 

    int rc(0); 

    stree a ; 
    rc = a.oldload(ss); 
    if( rc != 0 ) return rc ; 

    std::cout << "a.desc" << std::endl << a.desc_size() << std::endl ; 
    a.save("$FOLD") ;  

    stree b ; 
    rc = b.load("$FOLD"); 
    if( rc != 0 ) return rc ; 
 
    std::cout << "b.desc" << std::endl << b.desc_size() << std::endl ; 


    return 0 ; 
}
