#include <cassert>
#include <iostream>
#include <iomanip>
#include <array>

#include "SDice.hh"

int main(int argc, char** argv)
{
    SDice<26> rng ;

    std::array<unsigned, 26> hist ; 

    for(unsigned i=0 ; i < 1000000 ; i++) hist[rng()] += 1 ; 

    for(unsigned i=0 ; i < 26 ; i++) std::cout << std::setw(2) << i << " " << hist[i] << std::endl ; 

    return 0 ; 

}
