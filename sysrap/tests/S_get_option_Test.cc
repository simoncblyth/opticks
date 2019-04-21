#include <iostream>
#include "S_get_option.hh"

int main(int argc, char **argv)
{
    const char* size_default = "1200,800" ; 

    int stack = get_option<int>(argc, argv, "--stack", "3000" );   
    int width = get_option<int>(argc, argv, "--size,0", size_default ) ;   
    int height = get_option<int>(argc, argv, "--size,1", size_default ) ;   

    std::cout << " stack " << stack << std::endl ; 
    std::cout << " width [" << width << "]" << std::endl ; 
    std::cout << " height [" << height << "]" << std::endl ; 

    return 0 ; 
}


