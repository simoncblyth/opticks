// ./sfreq_test.sh

#include <cstdlib>
#include <iostream>
#include "sfreq.h"

const char* FOLD = getenv("FOLD"); 


void test_add_sort_save_load()
{
    sfreq c ; 

    c.add("red"); 
    c.add("green"); 
    c.add("blue"); 
    c.add("blue"); 
    c.add("blue"); 
    c.add("blue"); 
    c.add("red"); 

    c.sort(); 

    std::cout << c.desc() << std::endl ; 

    int n = c.get_freq("blue") ; 
    std::cout << " get_freq blue " << n << std::endl ; 

    c.save(FOLD); 


    sfreq c2 ; 
    c2.load(FOLD); 

    c2.set_disqualify("blue"); 
    assert( c2.is_disqualify("blue") ); 

    std::cout << "c2.desc\n" << c2.desc() << std::endl ; 
}

void test_empty_save_load()
{
    sfreq c ; 
    std::cout << "c.desc\n" << c.desc() << std::endl ; 
    c.save(FOLD); 

    sfreq c2 ; 
    c2.load(FOLD); 
    std::cout << "c2.desc\n" << c2.desc() << std::endl ; 
}



int main(int argc, char** argv)
{
    /*
    test_add_sort_save_load();
    */ 
    test_empty_save_load();

    return 0 ; 
}
