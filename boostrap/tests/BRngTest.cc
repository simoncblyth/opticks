#include <iostream>
#include "BRng.hh"

int main()
{

    BRng a ; 
    for(unsigned i=0 ; i < 10 ; i++ ) 
       std::cout << " a " << a() << std::endl ; 

    BRng b ; 
    for(unsigned i=0 ; i < 10 ; i++ ) 
       std::cout << " b " << b() << std::endl ; 


    return 0 ; 
}
