#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>

#include "OpticksUtil.hh"
#include "OpticksDebug.hh"
#include "NP.hh"


const char* KEY = "OPTICKS_RANDOM_SEQPATH" ;


void test_ExistsPath()
{
    bool x0 = OpticksUtil::ExistsPath("/tmp", "QCtxTest", "cerenkov_photon.npy" ); 
    bool x1 = OpticksUtil::ExistsPath("/tmp/QCtxTest/cerenkov_photon.npy" ); 
    assert( x0 == x1 ); 

    bool y0 = OpticksUtil::ExistsPath("/tmp", "QCtxTest" ); 
    bool y1 = OpticksUtil::ExistsPath("/tmp/QCtxTest" ); 
    assert( y0 == y1 ); 
}


void test_ListDir(const char* dir, const char* ext)
{
    std::vector<std::string> names ; 
    OpticksUtil::ListDir( names, dir, ext ); 

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

void test_Concatenate(const char* dir)
{ 
    std::cout << "[ test_Concatenate " << std::endl ; 

    NP* a = nullptr ; 

    const char* a_dir = dir ; 

    std::cout << "load from a_dir " << a_dir << std::endl ; 
    bool a_exists = OpticksUtil::ExistsPath(a_dir); 
    if(a_exists)
    {
        const char* ext = ".npy" ; 
        std::vector<std::string> names ; 
        OpticksUtil::ListDir( names, a_dir, ext ); 
        a = NP::Concatenate(a_dir, names); 
        std::cout << " a " << a->desc() << std::endl ; 
    }

    NP* b = nullptr ; 
    //const char* b_name = "rng_sequence_f_ni1000000_nj16_nk16_ioffset000000.npy" ; 
    const char* b_name   = "rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy" ; 
    bool b_exists = OpticksUtil::ExistsPath(dir, b_name);  
    if(b_exists)
    {
        std::cout << "load from b_name " << b_name  << std::endl ; 
        b = NP::Load(dir, b_name); 
        std::cout << " b " << b->desc() << std::endl ; 
    }

    if( a && b )
    {
        int cmp = NP::Memcmp(a, b); 
        std::cout << " NP::Memcmp(a, b) " << cmp << std::endl ; 
    }

    std::cout << "] test_Concatenate " << std::endl ; 
}




int main(int argc, char** argv)
{
    test_ExistsPath();  

    const char* dir = getenv(KEY); 
    if(dir == nullptr) std::cout << " missing KEY " << KEY << std::endl ; 
    if(dir == nullptr) return 0 ; 

    test_ListDir(dir, ".npy"); 

    test_Concatenate(dir); 

    return 0 ; 
}
