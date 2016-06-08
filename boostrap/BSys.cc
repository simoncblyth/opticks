#include "BSys.hh"
#include <iostream>


void BSys::WaitForInput(const char* msg)
{
    std::cerr << "BSys::WaitForInput " << msg << std::endl ; 
    char c = '\0' ;
    do
    {
        c = std::cin.get() ;  

    } while(c != '\n' ); 

   
    std::cerr << "BSys::WaitForInput DONE " << std::endl ; 

}
