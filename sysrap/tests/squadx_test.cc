// name=squadx_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>

#include "scuda.h"
#include "squad.h"

#include "squadx.h"
#include "stamp.h"
#include "NP.hh"

void test_quadx_0()
{
    quadx q[2] ; 

    q[1].w.x = stamp::Now(); 

    q[0].w.x = 0xffffffffeeeeeeee ; 
    q[0].w.y = 0xddddddddcccccccc ; 

    std::cout << "q[0].w.x " << std::hex << q[0].w.x << std::dec << std::endl ; 
    std::cout << "q[0].w.y " << std::hex << q[0].w.y << std::dec << std::endl ; 

    std::cout << "q[0].u.x " << std::hex << q[0].u.x << std::dec << std::endl ; 
    std::cout << "q[0].u.y " << std::hex << q[0].u.y << std::dec << std::endl ; 
    std::cout << "q[0].u.z " << std::hex << q[0].u.z << std::dec << std::endl ; 
    std::cout << "q[0].u.w " << std::hex << q[0].u.w << std::dec << std::endl ; 

    q[1].w.y = stamp::Now(); 


    NP* a = NP::Make<float>(2,4) ; 
    a->read2<float>( &q[0].f.x ); 
    a->save("$SQUADX_TEST_PATH") ; 
}


void test_quadx_1()
{
    quad4 q ; 
    q.zero(); 

    quadx4& qx = (quadx4&)q ; 
    qx.q3.w.x = stamp::Now(); 
    qx.q3.w.y = stamp::Now(); 


    NP* a = NP::Make<float>(4,4) ; 
    a->read2<float>( &qx.q0.f.x ); 
    a->save("$SQUADX_TEST_PATH") ; 
}



int main()
{
    /*
    test_quadx_0(); 
    */
    test_quadx_1(); 

    return 0 ; 
}
