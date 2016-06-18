#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

#include "BStr.hh"


int main(int, char**, char** envp)
{
    while(*envp) std::cout << *envp++ << std::endl ; 
 
    const char* path = getenv("PATH");
    typedef std::vector<std::string> VS ;
    VS elem ;  
    BStr::split(elem, path, ';');

    for(VS::const_iterator it=elem.begin() ; it != elem.end() ; it++) std::cout << *it << std::endl ;    

    return 0 ;
}
