#include "SMeta.hh"
#include "SPath.hh"
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



    bool create_dirs = true ; 
    const char* path = SPath::Resolve("$TMP", "SMetaTest.json", create_dirs ); 
    sm.save(path);

    std::cout << " sm " << std::endl << sm << std::endl ; 

    SMeta* smp = SMeta::Load(path); 

    std::cout << " smp " << std::endl << *smp << std::endl ; 
 

   return 0 ; 
}
