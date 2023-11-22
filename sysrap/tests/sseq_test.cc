// ~/opticks/sysrap/tests/sseq_test.sh 


#include <iostream>
#include <iomanip>
#include <bitset>

#include "scuda.h"
#include "squad.h"
#include "sseq.h"
#include "spath.h"

#include "NPX.h"
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



void test_desc_seqhis(const std::vector<unsigned>& history)
{
    sseq seq ; 
    seq.zero(); 

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

void test_desc_seqhis_0()
{
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

   test_desc_seqhis(history); 
}



template<int N>
std::string desc_flag(unsigned flag)
{
    std::stringstream ss ;
    ss 
       << std::hex << flag << std::dec 
       << " " 
       << std::bitset<N>(flag).to_string() 
       ; 
    std::string s = ss.str(); 
    return s ; 
}

template<typename T>
void test_wraparound(std::vector<unsigned>& history )
{
    unsigned long long seqhis = 0ull ; 
    unsigned long long four = 4ull ; 

    for(unsigned i=0 ; i < history.size() ; i++)
    {
        T flag = history[i]; 
        T slot = i ; 

        unsigned long long change = (( FFS(flag) & 0xfull ) << four*slot );

        seqhis |= change ;  

        std::cout 
            << std::setw(20) << OpticksPhoton::Flag(flag) 
            << " "
            << " FFS(flag) " << std::setw(1) << std::hex << FFS(flag) << std::dec
            << " "
            << " seqhis " << std::setw(16) << std::hex << seqhis << std::dec
            << " change " << std::setw(16) << std::hex << change << std::dec
            << std::endl
            ;
    }

    for(int i=0 ; i < 16 ; i++) std::cout << desc_flag<4>(i) << std::endl ; 
 
    std::cout << " (0xd | 0xc) == 0xd " << desc_flag<4>(0xd | 0xc) << std::endl ; 
    std::cout << " (0xc | 0xa) == 0xe " << desc_flag<4>(0xc | 0xa) << std::endl ; 
}


void test_desc_seqhis_1()
{
    std::cout << "test_desc_seqhis_1" << std::endl ; 
    std::vector<unsigned> history = { 
       TORCH,             // 0
       BOUNDARY_TRANSMIT, // 1
       BOUNDARY_TRANSMIT, // 2
       BOUNDARY_TRANSMIT, // 3
       BOUNDARY_TRANSMIT, // 4
       SURFACE_SREFLECT,  // 5
       SURFACE_SREFLECT,  // 6
       BOUNDARY_TRANSMIT, // 7
       BOUNDARY_REFLECT,  // 8
       BOUNDARY_REFLECT,  // 9
       BOUNDARY_TRANSMIT, // 10
       SURFACE_SREFLECT,  // 11
       SURFACE_SREFLECT,  // 12
       SURFACE_SREFLECT,  // 13
       BOUNDARY_TRANSMIT, // 14 
       BOUNDARY_REFLECT,  // 15 
       BOUNDARY_TRANSMIT, // 16  
       SURFACE_SREFLECT,  // 17
       BOUNDARY_TRANSMIT, // 18   
       SURFACE_ABSORB     // 19 
     } ; 

   test_desc_seqhis(history); 
   //test_wraparound<unsigned>(history); 
   //test_wraparound<unsigned long long>(history); 
}

/**

What happens when shifting beyond the width of the type is undefined.
But i see wraparound which causes overwriting. 

**/

void test_shiftwrap()
{
    unsigned long long seqhis = 0ull ; 
    for(unsigned j=0 ; j < 2 ; j++)
    for(unsigned i=0 ; i < 16 ; i++)
    {
        unsigned slot = j*16 + i ;  
        unsigned long long flag = j == 0 ? 0x5ull : 0xfull ;   
        unsigned long long change = flag << 4*slot ; 
        seqhis |= change ; 

        std::cout 
            << " seqhis " << std::setw(16) << std::hex << seqhis << std::dec
            << " change " << std::setw(16) << std::hex << change << std::dec
            << std::endl
            ;
    }
}

void test_load_seq()
{
    const char* path = spath::Resolve("$TMP/GEOM/$GEOM/G4CXTest/ALL0/p001/seq.npy") ; 
    NP* a = NP::Load(path); 
    std::cout << " path " << path << " a " << ( a ? a->sstr() : "-" ) << std::endl ; 

    std::vector<sseq> qq ; 
    NPX::VecFromArray<sseq>(qq, a ); 

    for(int i=0 ; i < std::min(int(qq.size()), 10) ; i++)
    {
        const sseq& q = qq[i] ; 
        std::cout << q.desc_seqhis() << std::endl ;  
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
    test_desc_seqhis_0();    
    test_desc_seqhis_1();    
    test_shiftwrap();    
    */

    test_load_seq(); 


    return 0 ; 
}
