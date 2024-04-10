// ~/o/sysrap/tests/SBitSet_test.sh

#include "SBitSet.h"

void test_value()
{
    SBitSet* bs = SBitSet::Create( 8, "VIZM" , "t" ); 

    std::cout << bs->desc() << std::endl ; 

    uint32_t val = bs->value<uint32_t>(); 
    std::cout << std::hex << val << std::dec << std::endl ;   
}


int main()
{
    uint32_t val = SBitSet::Value<uint32_t>( 8, "VIZM" , "t" );
    std::cout << std::hex << val << std::dec << std::endl ;   
    return 0 ; 
}  
