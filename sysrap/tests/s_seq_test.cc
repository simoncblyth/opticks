// ~/o/sysrap/tests/s_seq_test.sh

#include <iostream>
#include "s_seq.h"

int main()
{
    s_seq r ; 
    std::cout << r.desc() ; 

    r.setSequenceIndex(0) ; 
    std::cout << r.demo(10) << std::endl ;  

    r.setSequenceIndex(1) ; 
    std::cout << r.demo(10) << std::endl ;  

    // Returning to 0 continues the randoms in that stream (for a photon index)
    // they do not repeat unless the cursor is reset or cycled
    r.setSequenceIndex(0) ;   
    std::cout << r.demo(10) << std::endl ;  

    return 0 ; 
}; 

// ~/o/sysrap/tests/s_seq_test.sh

