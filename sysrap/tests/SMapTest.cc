
#include <string>
#include "SMap.hh"

#include "SYSRAP_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    SYSRAP_LOG__ ;

    typedef std::string K ; 
    typedef unsigned long long V ; 

    std::map<K,V> m ; 

    V v =  0xdeadbeefdeadbeefull ; 

    m["hello"] = v ; 
    m["world"] = v ; 
    m["other"] = 0x4full ; 
    m["yo"] = 0xffffull ; 


    unsigned nv = SMap<K,V>::ValueCount(m, v) ; 
    assert( nv == 2);

    bool dump = true ; 

    std::vector<K> keys ; 
    SMap<K,V>::FindKeys(m, keys, v, dump ) ; 

    assert( keys.size() == 2 ) ;


    return 0 ;
}
