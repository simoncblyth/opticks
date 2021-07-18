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
    const char* a_dir = "/tmp/QCtxTest/rng_sequence_f_ni1000000_nj16_nk16_tranche100000" ; 

    const char* ext = ".npy" ; 
    std::vector<std::string> names ; 
    OpticksDebug::ListDir( names, a_dir, ext ); 
    NP* a = NP::Concatenate(dir, names); 
    std::cout << " a " << a->desc() << std::endl ; 


    const char* bpath = "/tmp/QCtxTest/rng_sequence_f_ni1000000_nj16_nk16_tranche1000000/rng_sequence_f_ni1000000_nj16_nk16_ioffset000000.npy" ; 
    NP* b = NP::Load(bpath); 
    std::cout << " b " << b->desc() << std::endl ; 



}




int main(int argc, char** argv)
{
    return 0 ; 
}
