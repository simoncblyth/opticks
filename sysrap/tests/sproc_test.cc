// ./sproc_test.sh 

#include <iostream>
#include <iomanip>

#include "sproc.h"

void test_ExecutableName()
{
    std::cerr << sproc::ExecutableName() << std::endl ; 
}

void test_Query()
{
    uint32_t virtual_size_kb, resident_size_kb ; 
    for(int i=0 ; i < 100 ; i++)
    {
        sproc::Query(virtual_size_kb, resident_size_kb) ; 
        std::cout 
            << "  vsk " << std::setw(10) << virtual_size_kb 
            << "  kbf " << sproc::VirtualMemoryUsageKB()
            << "  mbf " << sproc::VirtualMemoryUsageMB()
            << "  rsk " << std::setw(10) << resident_size_kb 
            << "  kbf " << sproc::ResidentSetSizeKB()
            << "  mbf " << sproc::ResidentSetSizeMB()
            << std::endl 
            ;
    }
 
}



int main()
{
    test_Query(); 

    return 0 ; 
}
