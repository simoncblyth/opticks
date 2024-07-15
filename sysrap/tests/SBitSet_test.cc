// ~/o/sysrap/tests/SBitSet_test.sh

#include "SBitSet.h"

void test_value()
{
    SBitSet* bs = SBitSet::Create( 8, "VIZM" , "t" ); 

    std::cout << bs->desc() << std::endl ; 

    uint32_t val = bs->value<uint32_t>(); 
    std::cout << std::hex << val << std::dec << std::endl ;   
}


void test_Value()
{
    std::vector<std::string> specs = { 
              "t", 
              "t0", 
              "t0," , 
              "t0,1" , 
              "t0,1,2", 
              "t5",
               "",
               "0",
               "0,1",
               "0,1,2",
               "5,",
               "5",
              "16",
              "16," } ; 

    for(unsigned i=0 ; i < specs.size() ; i++)
    {
        const char* spec = specs[i].c_str() ; 
        uint32_t value = SBitSet::Value<uint32_t>( 32, spec );
        SBitSet* bs = SBitSet::Create<uint32_t>(value) ; 

        std::cout 
             << " spec " << std::setw(10) << spec 
             << " value " << std::setw(10) << std::hex << value << std::dec 
             << " bs " << bs->desc() 
             << "\n" 
             ; 
    }


}

void test_Create_value()
{
    uint32_t value = 0xdeadbeef ; 
    SBitSet* bs = SBitSet::Create<uint32_t>( value ); 
    std::cout << bs->desc() ; 
}


int main()
{
    //test_value(); 
    test_Value(); 
    //test_Create_value(); 

    return 0 ; 
}  
