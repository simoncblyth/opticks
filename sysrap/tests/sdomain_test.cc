// ./sdomain_test.sh 

#include <iostream>
#include "sdomain.h"

int main()
{
    std::cout << " sdomain::DomainLength() " << sdomain::DomainLength() << std::endl ;  

    sdomain dom ; 
    std::cout << dom.desc() << std::endl ; 

    NPFold* fold = dom.get_fold() ; 
    fold->save("$FOLD") ;    

    return 0 ; 
}

