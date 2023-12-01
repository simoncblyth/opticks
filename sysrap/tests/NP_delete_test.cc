// name=NP_delete_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include "stimer.h"
#include "sprof.h"
#include "NP.hh"

void test_delete_0()
{
    sprof p0, p1, p2 ; 

    const int M = 1000000 ; 
    const int K = 1000 ; 

    sprof::Stamp(p0);  

    NP* a = NP::Make<float>( 1*M, 4, 4 ) ; 
    std::cout << a->descSize() << std::endl ; 

    sprof::Stamp(p1);  

    //a->clear() ; 

    a->data.clear() ; 
    a->data.shrink_to_fit();
    //delete a ; 
    //a = nullptr ; 
    //stimer::sleep(1); 

    sprof::Stamp(p2);  


    std::cout << sprof::Desc(p0,p1) << std::endl ;  
    std::cout << sprof::Desc(p1,p2) << std::endl ;  
}



int main()
{
    test_delete_0(); 

    return 0 ; 
}

/**

NP::descSize arr_bytes 64000000 arr_kb 64000
398833,73441,64095
279619,0,8


**/

