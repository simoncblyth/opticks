// name=sseq_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

#include <iostream>

#include "scuda.h"
#include "squad.h"
#include "sseq.h"

#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"


void test_FFS_0()
{
    typedef long long LL ; 

    LL zero = 0ll ; 
    LL one  = 1ll ; 

    for(LL i=-1 ; i < 64 ; i++)
    { 
        LL x = i == -1 ? zero : ( one << i ) ;       
        std::cout 
            << " x " << std::setw(16) << std::hex << x << std::dec 
            << " FFS(x)  "  << std::setw(2) << FFS(x)  
            << " FFSLL(x) " << std::setw(2) << FFSLL(x)  
            << std::endl 
            ; 
    }
}

void test_FFS_1()
{
    typedef unsigned long long ULL ; 
    for(int i=-1 ; i < 64 ; i++)
    { 
        ULL x = i == -1 ? 0ull : ( 1ull << i ) ;       
        std::cout 
            << " x " << std::setw(16) << std::hex << x << std::dec 
            << " FFS(x)  "  << std::setw(2) << FFS(x)  
            << " FFSLL(x) " << std::setw(2) << FFSLL(x)  
            << std::endl 
            ; 
    }
}

void test_add_step_0()
{
    sseq seq ; 
    seq.zero(); 

    for(unsigned bounce=0 ; bounce < 16 ; bounce++)
    {
        unsigned flag = 0x1 << bounce ; 
        unsigned boundary = bounce ; 
        seq.add_step( bounce, flag, boundary ); 
        std::cout
            << " flag.dec " << std::setw(5) << std::dec << flag << std::dec 
            << " flag.hex " << std::setw(5) << std::hex << flag << std::dec 
            << " FFS(flag) " << std::setw(2) << FFS(flag) 
            << " ( FFS(flag) & 0xfull )  " << std::setw(2) << ( FFS(flag) & 0xfull ) 
            << " boundary " << std::setw(2) << boundary 
            << " ( boundary & 0xfull )  " << std::setw(2) << ( boundary & 0xfull ) 
            << seq.desc() 
            << std::endl 
            ; 
    }
    std::cout << "NB the nibble restriction means that FFS(flag) of 15 is the max step flag that can be carried in the seqhis sequence  " << std::endl ;  
}


void test_add_step_1()
{
    for(unsigned i=0 ; i < 16 ; i++)
    {
        unsigned flag = 0x1 << i ; 
        std::cout 
             << " flag " << std::setw(6) << flag 
             << " FFS(flag) " << std::setw(2) << FFS(flag) 
             << " OpticksPhoton::Flag(flag) " << std::setw(20) << OpticksPhoton::Flag(flag) 
             << std::endl
             ; 
    }

}




int main()
{
    test_FFS_0(); 
    test_FFS_1(); 
    test_add_step_0(); 
    test_add_step_1(); 

    return 0 ; 
}
