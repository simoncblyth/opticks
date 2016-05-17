#include "NPropNames.hpp"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <climits>

int main(int argc, char** argv)
{
    NPropNames pn("GMaterialLib");

    for(unsigned int i=0 ; i < pn.getNumLines() ; i++)
    {
        std::string line = pn.getLine(i) ;
        unsigned int index = pn.getIndex(line.c_str());

        std::cout << " i  " << std::setw(3) << i 
                  << " ix " << std::setw(3) << index
                  << " line " << line
                  << std::endl 
                  ; 

         assert(i == index);
    }

    assert( pn.getIndex("THIS_LINE_IS_NOT_PRESENT") == UINT_MAX );

    return 0 ; 
}
