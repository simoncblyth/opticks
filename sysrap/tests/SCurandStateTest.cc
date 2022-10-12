
#include <string>
#include <vector>
#include "OPTICKS_LOG.hh"
#include "SCurandState.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(info); 
    
    std::vector<std::string> specs = { "1:0:0", "3:0:0", "10:0:0" } ; 
    for(unsigned i=0 ; i < specs.size() ; i++)
    {
        const char* spec = specs[i].c_str(); 
        SCurandState scs(spec) ; 
        std::cout << scs.desc() << std::endl  ; 
    }

    return 0 ; 
}
