#include <cstdlib>
#include <iostream>
#include <limits>

#include "salloc.h"

const char* BASE = getenv("BASE"); 

void test_save_load()
{
    salloc* a = new salloc ; 
    a->add("one",   1, 0,0,0 ); 
    a->add("two",   2, 0,0,0 ); 
    a->add("three", 3, 0,0,0 ); 
    a->add("four",  4, 0,0,0 ); 
    a->set_meta<uint64_t>("max",  std::numeric_limits<uint64_t>::max()  ) ;  
    a->set_meta<uint64_t>("max2", 0xffffffffffffffff   ); 
    a->set_meta<uint64_t>("min",  std::numeric_limits<uint64_t>::min()  ) ;  

    std::cout << "a.desc" << std::endl << a->desc() ;     

    a->save(BASE); 

    salloc* b = salloc::Load(BASE)  ; 
    std::cout << "b.desc" << std::endl << b->desc() ;     
}

void test_load()
{
    salloc* a = salloc::Load(BASE) ; 
    std::cout << "a.desc" << std::endl << a->desc() ;     
}


int main(int argc, char** argv)
{
    test_save_load(); 
    return 0 ; 
}
