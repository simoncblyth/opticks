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


int main(int argc, char** argv)
{
    const char* dir = "/tmp/QCtxTest/rng_sequence_f" ; 
    const char* ext = ".npy" ; 
    std::vector<std::string> names ; 
    OpticksDebug::ListDir( names, dir, ext ); 
    NP* a = NP::Concatenate(dir, names); 

    return 0 ; 
}
