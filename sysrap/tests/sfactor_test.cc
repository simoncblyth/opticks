// ./sfactor_test.sh

#include <cstdlib>
#include <iostream>
#include <vector>
#include "sfactor.h"

#include "NP.hh"

const char* FOLD = getenv("FOLD") ; 
const char* name = "factors.npy" ; 

void Dump(const std::vector<sfactor>& factors)
{
    for(unsigned i=0 ; i < factors.size() ; i++ ) 
    {
        const sfactor& f = factors[i] ; 
        std::cout << f.desc() << std::endl ; 
    }
}

void test_Write()
{
    std::vector<sfactor> factors ;     
    for(unsigned i=0 ; i < 10 ; i++)
    {
        sfactor f ; 
        f.index = i ; 
        f.set_sub("0123456789abcdef0123456789abcdef"); 
        f.freq = 100*i ; 
        factors.push_back(f) ;  
    }
    Dump(factors); 

    NP::Write<int>(FOLD, name, (int*)factors.data(), factors.size(), sfactor::NV ); 
    std::cout << "wrote " << FOLD << "/" << name  << std::endl ; 
}

void test_Read()
{
    std::vector<sfactor> factors ;     
    NP* a = NP::Load(FOLD, name); 
    factors.resize(a->shape[0]); 
    memcpy( (int*)factors.data(), a->bytes(), a->arr_bytes() ); 
    Dump(factors); 
}


int main(int argc, char** argv)
{
    test_Read(); 
    return 0 ; 
}
