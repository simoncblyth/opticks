/**
salloc_test.cc
==============

~/o/sysrap/tests/salloc_test.sh 

**/

#include <cstdlib>
#include <iostream>
#include <limits>

#include "ssys.h"
#include "salloc.h"

const char* BASE = getenv("BASE"); 

struct salloc_test
{
    static int save_load(); 
    static int load(); 
    static int main(); 
};

int salloc_test::save_load()
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

    return 0 ; 
}

int salloc_test::load()
{
    salloc* a = salloc::Load(BASE) ; 
    std::cout << "a.desc" << std::endl << a->desc() ;     
    return 0 ; 
}

int salloc_test::main()
{
    const char* TEST = ssys::getenvvar("TEST","save_load"); 
    int rc = 0 ; 
    if(strcmp(TEST,"save_load")==0 ) rc += save_load(); 
    if(strcmp(TEST,"load")==0 )      rc += load(); 
    return rc ; 
}

int main(){ return salloc_test::main() ; }
    
