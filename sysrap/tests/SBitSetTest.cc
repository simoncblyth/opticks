#include "SBitSet.hh"
#include "OPTICKS_LOG.hh"

const char* SPECS = R"LITERAL(
1,10,100,-1
~1,10,100,-1
t1,10,100,-1
t
~
~0
t0
t-1
~-1
0

0,1,2,3
~0,1,2,3
)LITERAL" ; 

void test_0()
{
    LOG(info); 

    unsigned num_bits = 128 ; 
    bool* bits = new bool[num_bits] ; 

    std::stringstream ss(SPECS) ;    
    std::string line ; 
    while (std::getline(ss, line))  
    {   
        const char* spec = line.c_str(); 
        SBitSet::Parse( num_bits, bits, spec ); 
        std::cout 
            << std::setw(20) << spec 
            << " : "
            << SBitSet::Desc(num_bits, bits,  false )
            << std::endl 
            ; 
    }    
}

void test_1()
{
    LOG(info); 

    SBitSet* bs = new SBitSet(128); 

    std::stringstream ss(SPECS) ;    
    std::string line ; 
    while (std::getline(ss, line))  
    {   
        const char* spec = line.c_str(); 

        bs->parse(spec); 

        std::cout 
            << std::setw(20) << spec 
            << " : "
            << bs->desc()
            << std::endl 
            ; 

    }  

    delete bs ; 
}


void test_2()
{
    LOG(info); 
    std::stringstream ss(SPECS) ;    
    std::string line ; 
    while (std::getline(ss, line))  
    {   
        const char* spec = line.c_str(); 
        SBitSet* bs = SBitSet::Create(128, spec); 
        std::cout 
            << std::setw(20) << spec 
            << " : "
            << bs->desc()
            << std::endl 
            ; 

        delete bs ; 
    }  
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_0(); 
    test_1(); 
    test_2(); 

    return 0 ; 
}
