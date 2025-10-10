// ~/o/sysrap/tests/SBitSet_test.sh

#include "SBitSet.h"

struct SBitSet_test
{
    static int value();
    static int Value();
    static int Create();
    static int Roundtrip();
    static int Main();
};


int SBitSet_test::value()
{
    SBitSet* bs = SBitSet::Create( 8, "VIZM" , "t" );

    std::cout << bs->desc() << std::endl ;

    uint32_t val = bs->value<uint32_t>();
    std::cout << std::hex << val << std::dec << std::endl ;

    return 0 ;
}


int SBitSet_test::Value()
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
    return 0 ;
}

int SBitSet_test::Create()
{
    uint32_t value = 0xdeadbeef ;
    SBitSet* bs = SBitSet::Create<uint32_t>( value );
    std::cout << bs->desc() ;
    return 0 ;
}

int SBitSet_test::Roundtrip()
{
    uint32_t value = 0xdeadbeef ;
    SBitSet* bs0 = SBitSet::Create<uint32_t>( value );

    std::vector<unsigned char> bytes ;
    bs0->serialize(bytes);

    SBitSet* bs1 = SBitSet::CreateFromBytes(bytes);
    int cmp = bs0->compare(bs1);

    std::cout
       << "SBitSet_test::Roundtrip"
       << "\n"
       << " bs0 " << bs0->desc() << "\n"
       << " bs1 " << bs1->desc() << "\n"
       << " bytes.size " << bytes.size()
       << " cmp " << cmp
       << "\n"
       ;

    return cmp ;
}




int SBitSet_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST","Create");
    bool all = strcmp(TEST,"ALL") == 0 ;
    int rc = 0 ;
    if(all||0==strcmp(TEST,"value")) rc += value();
    if(all||0==strcmp(TEST,"Value")) rc += Value();
    if(all||0==strcmp(TEST,"Create")) rc += Create();
    if(all||0==strcmp(TEST,"Roundtrip")) rc += Roundtrip();

    return rc ;
}

int main()
{
    return SBitSet_test::Main();
}
