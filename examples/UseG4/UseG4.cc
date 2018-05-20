#include <iostream>
#include "G4Version.hh"


void UseG4_dump_version()
{
    std::cout << "G4VERSION_NUMBER " << G4VERSION_NUMBER << std::endl ; 
    std::cout << "G4VERSION_TAG    " << G4VERSION_TAG << std::endl ; 
    std::cout << "G4Version        " << G4Version << std::endl ; 
    std::cout << "G4Date           " << G4Date << std::endl ; 
}


#ifdef WITH_MAIN
int main()
{
    UseG4_dump_version();
    return 0 ; 
}
#endif
