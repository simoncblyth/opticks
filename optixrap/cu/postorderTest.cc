// clang++ postorderTest.cc -lc++ -o $TMP/postorderTest && $TMP/postorderTest

#include <iostream>
#include <iomanip>
#include "postorder.h"

int main()
{
    unsigned height = 2 ; 
    unsigned nodeIdx = 1 << height ; 
    unsigned index = 0 ; 

    while( nodeIdx != 0)
    {
        unsigned depth = POSTORDER_DEPTH( nodeIdx );
        unsigned elevation = height - depth ; 
     
        std::cout 
                  << " index " << std::setw(3) << index
                  << " nodeIdx " << std::setw(3) << nodeIdx  
                  << " height " << height 
                  << " depth " << depth 
                  << " elevation " << elevation
                  << std::endl 
                  ;

        nodeIdx = POSTORDER_NEXT( nodeIdx, elevation );
        index++ ; 
    }


    return 0 ;
}
