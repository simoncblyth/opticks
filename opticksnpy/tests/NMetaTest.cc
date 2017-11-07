
#include <string>

#include "NMeta.hpp"
#include "NPY_LOG.hh"
#include "PLOG.hh"

void test_composable()
{
    const char* path = "$TMP/NMetaTest/test_composable.json" ;

    NMeta m1 ; 
    m1.set<int>("red", 1);
    m1.set<int>("green", 2);
    m1.set<int>("blue", 3);
    m1.set<float>("pi", 3.1415);
    m1.set<std::string>("name", "yo");

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


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    //test_write_read();
    test_composable();

    return 0 ; 
}

