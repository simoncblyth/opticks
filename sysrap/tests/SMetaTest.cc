#include "SMeta.hh"
#include "SPath.hh"
#include "OPTICKS_LOG.hh"

void Populate(SMeta& sm )
{
    double dval = 1.4124 ; 
    const char* key = "key" ; 
    std::string s = "string" ; 

    sm.js["red"] = "cyan" ; 
    sm.js["green"] = "magenta" ; 
    sm.js["blue"] = "yellow" ; 
    sm.js["dval"] = dval ; 

    sm.js[key] = s ; 
}


void test_1()
{
    SMeta sm ; 
    Populate(sm); 

    bool create_dirs = 1 ; // 1:filepath
    const char* path = SPath::Resolve("$TMP", "SMetaTest.json", create_dirs ); 
    LOG(info) << path ; 
    sm.save(path);
    std::cout << " sm " << std::endl << sm << std::endl ; 
    SMeta* smp = SMeta::Load(path); 
    std::cout << " smp " << std::endl << *smp << std::endl ; 
}


void test_2()
{
    SMeta sm ; 
    Populate(sm); 

    int create_dirs = 1 ; // 1::filepath 
    const char* path = SPath::Resolve("$TMP/red/green/blue", "SMetaTest.json", create_dirs); 
    LOG(info) << path ; 

    sm.save(path);
    std::cout << " sm " << std::endl << sm << std::endl ; 
    SMeta* smp = SMeta::Load(path); 
    std::cout << " smp " << std::endl << *smp << std::endl ; 
}





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    LOG(info) ; 

    test_2(); 

    return 0 ; 
}
