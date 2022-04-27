// name=sphoton_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

#include <iostream>

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"


int main()
{
    qphoton qp ; 
    qp.q.zero(); 

    std::cout << qp.q.desc() << std::endl ; 
    return 0 ; 
}
