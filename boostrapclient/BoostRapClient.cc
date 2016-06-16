#include <iostream>

//#include "BTime.hh"
#include "BDemo.hh"

int main(int argc, char** argv)
{
    std::cerr << " argc " << argc 
              << " argv[0] " << argv[0] 
              << std::endl ; 

    //BTime bt ; 
    //std::cerr << bt.check() << std::endl ; 
    //std::cout << BTime::now("%Y",0) << std::endl ; 

    BDemo bd(42);
    bd.check();

    std::cerr << "checked" << std::endl ; 


    return 0 ; 
}
