#include "BTime.hh"

#include <iostream>


int main(int argc, char** argv)
{
    BTime bt ; 
    std::cout << " argc " << argc 
              << " argv[0] " << argv[0]
              << " check " << bt.check()
              << " now " << bt.now("%Y",0)
              << std::endl ; 



}
