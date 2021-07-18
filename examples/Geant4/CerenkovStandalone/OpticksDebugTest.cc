#include <string>
#include <vector>
#include <iostream>

#include "OpticksDebug.hh"
#include "NP.hh"

void test_ListDir()
{
    const char* dir = "/tmp/QCtxTest/rng_sequence_f" ; 
    const char* ext = ".npy" ; 
    std::vector<std::string> names ; 
    OpticksDebug::ListDir( names, dir, ext ); 

    std::cout 
        << "OpticksDebug::ListDir" 
        << " dir " << dir 
        << " ext " << ext 
        << " names.size " << names.size() 
        << std::endl 
        ; 

    for(unsigned i=0 ; i < names.size() ; i++)
    {
        std::cout << names[i] << std::endl ; 
    }
}

void test_Concatenate()
{ 
    std::cout << "[ test_Concatenate " << std::endl ; 

    NP* a = nullptr ; 
    const char* a_dir = "/tmp/QCtxTest/rng_sequence_f_ni1000000_nj16_nk16_tranche100000" ; 
    std::cout << "load from a_dir " << a_dir << std::endl ; 
    bool a_exists = OpticksDebug::ExistsPath(a_dir); 
    if(a_exists)
    {
        const char* ext = ".npy" ; 
        std::vector<std::string> names ; 
        OpticksDebug::ListDir( names, a_dir, ext ); 
        a = NP::Concatenate(a_dir, names); 
        std::cout << " a " << a->desc() << std::endl ; 
    }

    NP* b = nullptr ; 
    const char* b_path = "/tmp/QCtxTest/rng_sequence_f_ni1000000_nj16_nk16_tranche1000000/rng_sequence_f_ni1000000_nj16_nk16_ioffset000000.npy" ; 
    bool b_exists = OpticksDebug::ExistsPath(b_path);  
    if(b_exists)
    {
        std::cout << "load from b_path " << b_path  << std::endl ; 
        b = NP::Load(b_path); 
        std::cout << " b " << b->desc() << std::endl ; 
    }

    if( a && b )
    {
        int cmp = NP::Memcmp(a, b); 
        std::cout << " NP::Memcmp(a, b) " << cmp << std::endl ; 
    }

    std::cout << "] test_Concatenate " << std::endl ; 
}

void test_ExistsPath()
{
    bool x0 = OpticksDebug::ExistsPath("/tmp", "QCtxTest", "cerenkov_photon.npy" ); 
    bool x1 = OpticksDebug::ExistsPath("/tmp/QCtxTest/cerenkov_photon.npy" ); 
    assert( x0 == x1 ); 

    bool y0 = OpticksDebug::ExistsPath("/tmp", "QCtxTest" ); 
    bool y1 = OpticksDebug::ExistsPath("/tmp/QCtxTest" ); 
    assert( y0 == y1 ); 
}


int main(int argc, char** argv)
{
    test_Concatenate(); 
    test_ExistsPath();  
    return 0 ; 
}
