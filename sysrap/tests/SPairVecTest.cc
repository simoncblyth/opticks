#include "SPairVec.hh"

#include <string>

#include "SYSRAP_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    SYSRAP_LOG__ ;

    typedef std::string K ; 
    typedef unsigned    V ; 
    typedef std::pair<K,V> PKV ; 
    typedef std::vector<PKV> LPKV ; 

    LPKV lpkv ; 
    lpkv.push_back(PKV("red", 100)) ; 
    lpkv.push_back(PKV("green", 10)) ; 
    lpkv.push_back(PKV("bule", 50)) ; 

    SPairVec<K,V> spv(lpkv, false ); 
    spv.dump("bef");
    spv.sort();
    spv.dump("aft");


    return 0 ;
}
