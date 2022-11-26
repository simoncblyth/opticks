// name=spho_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include "spho.h"

int main()
{
     spho p = {} ; 
     p.id = 101 ; 
     std::cout << p.desc() << std::endl ;  

     for(unsigned i=0 ; i < 256 ; i++)
     {
         if( i == 10 ) p.uc4.w = 214 ;  

         p = p.make_nextgen(); 
         std::cout << p.desc() << std::endl ;  
     }



     p.set_gen(1); 
     std::cout 
         << p.desc() << " "
         << " p.set_gen(1) " 
         << std::endl 
         ;  

     p.set_gen(0); 
     p.set_flg(1);  

     std::cout 
         << p.desc() << " " 
         << "p.set_gen(0), p.set_flg(1) " 
         << std::endl 
         ;  


     p.set_gen(1); 
     p.set_flg(1);  
     std::cout 
         << p.desc() << " " 
         << "p.set_gen(1), p.set_flg(1) " 
         << std::endl 
         ;  

     p.uc4.x = 1 ; 
     p.uc4.y = 1 ; 
     p.uc4.z = 1 ; 
     p.uc4.w = 1 ; 
     std::cout 
         << p.desc() << " " 
         << "p.uc4.x = 1, p.uc4.y = 1, p.uc4.z = 1, p.uc4.w = 1 " 
         << std::endl 
         ;  



     p.uc4.x = 0xff ; 
     p.uc4.y = 0xff ; 
     p.uc4.z = 0xff ; 
     p.uc4.w = 0xff ; 
     std::cout 
         << p.desc() << " " 
         << "p.uc4.x = 0xff, p.uc4.y = 0xff, p.uc4.z = 0xff, p.uc4.w = 0xff " 
         << std::endl 
         ;  


     return 0 ; 
}
