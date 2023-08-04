#include "NPFold.h"

int main()
{
    const char* path = "$PMTSimParamData_BASE" ; 
    NPFold* jpmt = NPFold::Exists(path) ? NPFold::Load(path) : nullptr ; 
    std::cout << path << std::endl << ( jpmt ? jpmt->desc() : "-" ) << std::endl ; 

    return 0 ; 
}
