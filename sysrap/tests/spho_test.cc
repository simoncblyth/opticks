// name=spho_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include "spho.h"

int main()
{
     spho p = {} ; 
     p.id = 101 ; 
     std::cout << p.desc() << std::endl ;  

     for(unsigned i=0 ; i < 10 ; i++)
     {
         p = p.make_reemit(); 
         std::cout << p.desc() << std::endl ;  
     }


     return 0 ; 
}
