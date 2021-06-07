
#include <vector>
#include <string>

#include "OPTICKS_LOG.hh"
#include "SSys.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    std::vector<std::string> keys = {"USER", "HOME", "USER,HOME", "HOME,USER" } ; 


    for(unsigned i=0 ; i < keys.size() ; i++)
    {
         const char* key = keys[i].c_str(); 
         const char* val = SSys::getenvvar(key) ; 
         LOG(info)
            << std::setw(25) << key 
            << " : "
            << std::setw(25) << val
            ;
    }
   

    return 0 ; 
}
