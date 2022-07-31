// ./sfreq_test.sh

#include <cstdlib>
#include <iostream>
#include "sfreq.h"

const char* FOLD = getenv("FOLD"); 

int main(int argc, char** argv)
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

    c2.set_disqualified("blue"); 
    assert( c2.is_disqualified("blue") ); 

    std::cout << "c2.desc\n" << c2.desc() << std::endl ; 


    return 0 ; 
}
