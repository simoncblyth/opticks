#include "NPFold.h"

int main()
{
    NPFold* jpmt = NPFold::Load("$PMTSimParamData_BASE") ; 
    std::cout << ( jpmt ? jpmt->desc() : "-" ) << std::endl ; 

    return 0 ; 
}
