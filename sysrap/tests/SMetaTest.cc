#include "SMeta.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(info) ; 

    SMeta sm ; 

    double dval = 1.4124 ; 
    const char* key = "key" ; 
    std::string s = "string" ; 

    sm.js["red"] = "cyan" ; 
    sm.js["green"] = "magenta" ; 
    sm.js["blue"] = "yellow" ; 
    sm.js["dval"] = dval ; 

    sm.js[key] = s ; 




    const char* dir = "$TMP" ; 
    const char* name = "SMetaTest.json" ; 
    sm.save(dir, name);

    std::cout << " sm " << std::endl << sm << std::endl ; 

    SMeta* smp = SMeta::Load(dir, name); 

    std::cout << " smp " << std::endl << *smp << std::endl ; 
 

   return 0 ; 
}
