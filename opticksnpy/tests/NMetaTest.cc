
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

    NMeta m2 ; 
    m2.set<int>("cyan", 7);
    m2.set<int>("red", 100);
    m2.set<int>("green", 200);
    m2.set<int>("blue", 300);
    m2.set<float>("pi", 3.1415);
    m2.set<std::string>("name", "yo");

    NMeta m ;
    m.set("m1", &m1 );
    m.set("m2", &m2 );

    m.dump();  
    m.save(path);
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

