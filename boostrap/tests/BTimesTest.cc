// TEST=BTimesTest om-t

#include "BTimes.hh"

#include <iostream>
#include <vector>
#include <cstdlib>

#include "OPTICKS_LOG.hh"


void test_load_and_compare()
{
    LOG(info); 
    if(getenv("IDPATH")==NULL)
    {
        LOG(info) << "missing envvar IDPATH" ; 
        return ; 
    }

    BTimes* ck = BTimes::load("ck", "$IDPATH/times", "cerenkov_1.ini");
    BTimes* sc = BTimes::load("sc", "$IDPATH/times", "scintillation_1.ini") ;
    BTimes* cks = ck->clone("cks");
    cks->setScale( 2817543./612841. );   // scale up according to photon count 

    std::vector<BTimes*> vt ; 
    vt.push_back(ck);
    vt.push_back(cks);
    vt.push_back(sc);

    BTimes::compare(vt);
}


void test_create_and_compare()
{
    LOG(info); 
    BTimes* a = new BTimes("a") ; 
    BTimes* b = new BTimes("b") ; 
    BTimes* c = new BTimes("c") ; 
    
    for(int i=-5 ; i < 6 ; i++ )
    {
        a->add("atest", i, double(i)*1. ) ; 
        b->add("atest", i, double(i)*10. ) ; 
        c->add("atest", i, double(i)*100. ) ; 
    }

    //a->dump("a"); 
    //b->dump("b"); 
    //c->dump("c"); 

    std::vector<BTimes*> vt = { a, b, c } ; 
    vt.push_back(a);
    vt.push_back(b);
    vt.push_back(c);

    BTimes::compare(vt); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_load_and_compare();
    test_create_and_compare();

    return 0 ; 
}
