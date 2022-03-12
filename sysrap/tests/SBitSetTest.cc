#include "SBitSet.hh"
#include "OPTICKS_LOG.hh"

const char* SPECS = R"LITERAL(
1,10,100,-1
~1,10,100,-1
t1,10,100,-1
)LITERAL" ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    unsigned num_bits = 128 ; 
    bool* bits = new bool[num_bits] ; 

    std::stringstream ss(SPECS) ;    
    std::string line ; 
    while (std::getline(ss, line))  
    {   
        if(line.empty()) continue ;   
        const char* spec = line.c_str(); 
        SBitSet::Parse( bits, num_bits, spec ); 
        std::cout 
            << std::setw(20) << spec 
            << " : "
            << SBitSet::Desc(bits, num_bits, false )
            << std::endl 
            ; 
    }    

    return 0 ; 
}
