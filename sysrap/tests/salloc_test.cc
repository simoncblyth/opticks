#include <cstdlib>
#include <iostream>
#include "salloc.h"

const char* BASE = getenv("BASE"); 

void test_save_load()
{
    salloc a ; 
    a.add("one",   1, 0,0,0 ); 
    a.add("two",   2, 0,0,0 ); 
    a.add("three", 3, 0,0,0 ); 
    a.add("four",  4, 0,0,0 ); 
    std::cout << "a.desc" << std::endl << a.desc() ;     
    a.save(BASE); 

    salloc b  ; 
    b.load(BASE); 
    std::cout << "b.desc" << std::endl << b.desc() ;     
}


int main(int argc, char** argv)
{
    salloc a ; 
    a.load(BASE); 
    std::cout << "a.desc" << std::endl << a.desc() ;     
    return 0 ; 
}
