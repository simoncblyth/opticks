// TEST=NMetaTest om-t

#include <string>

#include "NMeta.hpp"
#include "OPTICKS_LOG.hh"

void test_composable()
{
    const char* path = "$TMP/NMetaTest/test_composable.json" ;

    NMeta m1 ; 
    assert( m1.size() == 0 ); 

    m1.set<int>("red", 1);
    assert( m1.size() == 1 ); 
    m1.set<int>("green", 2);
    m1.set<int>("blue", 3);
    m1.set<float>("pi", 3.1415);
    m1.set<std::string>("name", "yo");

    assert( m1.size() == 5 ); 
    assert( m1.getNumKeys() == 5 ); 

    NMeta m2 ; 
    m2.set<int>("cyan", 7);
    m2.set<int>("red", 100);
    m2.set<int>("green", 200);
    m2.set<int>("blue", 300);
    m2.set<float>("pi", 3.1415);
    m2.set<std::string>("name", "yo");

    assert( m2.getNumKeys() == 6 ); 

    NMeta m ;
    m.setObj("m1", &m1 );
    m.setObj("m2", &m2 );
    m.setObj("m3", &m2 );
    m.setObj("m4", &m2 );

    unsigned xkey = 4 ; 
    assert( m.getNumKeys() == xkey ); 

    m.dump();  
    m.save(path);

    NMeta* ml = NMeta::Load(path);
    ml->dump();

    unsigned num_key = ml->getNumKeys() ;
    assert( num_key == xkey );

    for(unsigned i=0 ; i < num_key ; i++)
    {
        const char* key = ml->getKey(i);
        NMeta* lm = ml->getObj(key);
        LOG(info) 
            << " i " << i 
            << " key " << key 
            << " lm " << lm->desc()
            ; 
    }
}

void test_write_read()
{
    const char* path = "$TMP/NMetaTest/test_write_read.json" ;

    NMeta m ; 
    m.set<int>("red", 1);
    m.set<int>("green", 2);
    m.set<int>("blue", 3);
    m.set<float>("pi", 3.1415);
    m.set<std::string>("name", "yo");

    m.save(path);
    m.dump();  
}

void test_copy_ctor()
{
    NMeta m ; 
    m.set<int>("red", 1);
    m.set<int>("green", 2);
    m.set<int>("blue", 3);
    m.set<float>("pi", 3.1415);
    m.set<std::string>("name", "yo");

    assert( m.getNumKeys() == 5 );

    m.dump();  

    NMeta mc(m);  
    mc.dump("copy-ctor");

    assert( mc.getNumKeys() == 5 );
}
 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_write_read();
    test_composable();
    test_copy_ctor();

    return 0 ; 
}

