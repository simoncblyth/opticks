// name=sseq_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>
#include <bitset>

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

        unsigned f32 = FFS(x) ;     
        unsigned f64 = FFSLL(x) ;     

        unsigned x32 = rFFS(f32) ; 
        ULL x64 = rFFSLL(f64) ; 

        std::cout 
            // << " b " << std::setw(64) << std::bitset<64>(x) 
            << " i " << std::setw(3) << i 
            << " x " << std::setw(16) << std::hex << x << std::dec 
            << " FFS(x)  "  << std::setw(2) << f32
            << " FFSLL(x) " << std::setw(2) << f64
            << " x32 " << std::setw(16) << std::hex << x32 << std::dec 
            << " x64 " << std::setw(16) << std::hex << x64 << std::dec 
            << std::endl 
            ; 

        if(i < 32) assert( x32 == x ); 
        assert( x64 == x ); 
    }
}

void test_add_nibble_0()
{
    sseq seq ; 
    seq.zero(); 

    for(unsigned bounce=0 ; bounce < 16 ; bounce++)
    {
        unsigned flag = 0x1 << bounce ; 
        unsigned boundary = bounce ; 
        seq.add_nibble( bounce, flag, boundary ); 
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


void test_add_nibble_1()
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

void test_GetNibble()
{
    typedef unsigned long long ULL ; 
    ULL x = 0x0123456789abcdefull ; 

    for(unsigned i=0 ; i < 16 ; i++) 
    {
        unsigned nib = sseq::GetNibble(x, i); 
        std::cout << std::setw(3) << i << " nib " << std::hex << nib << std::dec << std::endl ; 
    }
}

void test_ClearNibble()
{
    typedef unsigned long long ULL ; 
    ULL x = 0xa123456789abcdefull ; 

    for(int i=-1 ; i < 16 ; i++) 
    {
        if( i > -1 ) sseq::ClearNibble(x, i); 
        std::cout << std::setw(3) << i << " x " << std::setw(16) << std::setfill('0') << std::hex << x << std::dec << std::endl ; 
    }
}

void test_SetNibble()
{
    typedef unsigned long long ULL ; 
    ULL x = 0xa123456789abcdefull ; 

    for(int i=-1 ; i < 16 ; i++) 
    {
        if( i > -1 ) sseq::SetNibble(x, i, 0xf); 
        std::cout << std::setw(3) << i << " x " << std::setw(16) << std::setfill('0') << std::hex << x << std::dec << std::endl ; 
    }
}

void test_get_flag_set_flag()
{
    sseq seq ; 
    seq.zero(); 

    std::vector<unsigned> history = { 
       CERENKOV, 
       BOUNDARY_TRANSMIT, 
       BOUNDARY_TRANSMIT, 
       BULK_SCATTER, 
       BULK_REEMIT, 
       BOUNDARY_TRANSMIT, 
       SURFACE_DETECT,
       BULK_ABSORB,
       SCINTILLATION,
       TORCH
     } ; 

    for(unsigned i=0 ; i < history.size() ; i++) seq.add_nibble(i, history[i], 0) ; 

    //std::cout << OpticksPhoton::FlagSequence(seq.seqhis) << std::endl ; 
    std::cout << seq.desc_seqhis() << std::endl ;  

    for(unsigned i=0 ; i < history.size() ; i++) 
    {
        unsigned flag = seq.get_flag(i) ; 
        assert( flag == history[i] ); 
        std::cout << OpticksPhoton::Flag(flag) << std::endl ; 
        std::cout << seq.desc_seqhis() << std::endl ;  
    }

    //std::cout << OpticksPhoton::FlagSequence(seq.seqhis) << std::endl ; 
    std::cout << seq.desc_seqhis() << std::endl ;  

    for(unsigned i=0 ; i < history.size() ; i++) 
    {
        unsigned flag = seq.get_flag(i) ; 
        if(flag == BULK_ABSORB) seq.set_flag(i, BULK_REEMIT) ; 
    }

    //std::cout << OpticksPhoton::FlagSequence(seq.seqhis) << std::endl ; 
    std::cout << seq.desc_seqhis() << std::endl ;  


}

void test_desc_seqhis()
{
    sseq seq ; 
    seq.zero(); 

    std::vector<unsigned> history = { 
       CERENKOV, 
       BOUNDARY_TRANSMIT, 
       BOUNDARY_TRANSMIT, 
       BULK_SCATTER, 
       BULK_REEMIT, 
       BOUNDARY_TRANSMIT, 
       SURFACE_DETECT,
       BULK_ABSORB,
       SCINTILLATION,
       TORCH,
       BOUNDARY_TRANSMIT, 
       BOUNDARY_TRANSMIT, 
       BULK_SCATTER, 
       BULK_REEMIT, 
       BOUNDARY_TRANSMIT, 
       SURFACE_DETECT,
       BULK_ABSORB,
       SCINTILLATION
     } ; 

     for(int bounce=0 ; bounce < history.size() ; bounce++)
     {
        unsigned flag = history[bounce] ; 
        unsigned boundary = 0 ;  
        seq.add_nibble( bounce, flag, boundary ); 

        std::cout 
            << std::setw(20) << OpticksPhoton::Flag(flag) 
            << " : " <<  seq.desc_seqhis() 
            << std::endl 
            ; 
     }
}



int main()
{
    /*
    test_FFS_0(); 
    test_FFS_1(); 
    test_add_nibble_0(); 
    test_add_nibble_1(); 
    test_GetNibble(); 
    test_ClearNibble(); 
    test_SetNibble(); 
    test_get_flag_set_flag(); 
    */

    test_desc_seqhis();    

    return 0 ; 
}
