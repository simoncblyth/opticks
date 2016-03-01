#include "NPropNames.hpp"

#include <iostream>
#include <iomanip>

int main(int argc, char** argv)
{
    NPropNames pn("GMaterialLib");

    for(unsigned int i=0 ; i < pn.getNumLines() ; i++)
    {
        std::cout << std::setw(3) << i 
                  << " " << pn.getLine(i)
                  << std::endl 
                  ; 
    }

    return 0 ; 
}
