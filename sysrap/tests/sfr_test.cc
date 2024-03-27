/**

~/o/sysrap/tests/sfr_test.sh 
~/o/sysrap/tests/sfr_test.cc
~/o/sysrap/tests/sfr_test.py 

**/

#include "sfr.h"

int main()
{
    sfr a ; 
    a.set_name("hello a"); 

    a.aux0.x = 1 ; 
    a.aux0.y = 2 ; 
    a.aux0.z = 3 ; 
    a.aux0.w = 4 ;

    a.aux1.x = 10 ; 
    a.aux1.y = 20 ; 
    a.aux1.z = 30 ; 
    a.aux1.w = 40 ;

    a.aux2.x = 100 ; 
    a.aux2.y = 200 ; 
    a.aux2.z = 300 ; 
    a.aux2.w = 400 ;


    std::cout << "A\n" << a.desc() << std::endl; 
    a.save("$FOLD"); 

    sfr b = sfr::Load("$FOLD"); 
    std::cout << "B\n" << b.desc() << std::endl; 

    return 0 ; 
}
