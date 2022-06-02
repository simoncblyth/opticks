// name=NPFold_test ; mkdir -p /tmp/$name ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name/$name && /tmp/$name/$name

#include "NPFold.h"

void test_0()
{
    NP* a = NP::Make<float>(1,4,4) ; 
    a->fillIndexFlat(); 

    NP* b = NP::Make<float>(1,4,4) ; 
    b->fillIndexFlat(); 

    NPFold nf0 ; 
    nf0.add("a.npy", a ); 
    nf0.add("some/relative/path/a.npy", a ); 
    nf0.add("b.npy", b ); 

    std::cout << "nf0" << std::endl << nf0.desc()  ; 

    const char* base = "/tmp/NPFold_test/base" ; 
    nf0.save(base); 

    NPFold* nf1 = NPFold::Load(base); 

    std::cout << "nf1" << std::endl << nf1->desc()  ; 


    int cf = NPFold::Compare(&nf0, nf1, true ); 
    assert( cf == 0 ); 
}

void test_add_without_ext()
{
    NP* a = NP::Make<float>(1,4,4) ; 
    a->fillIndexFlat(); 

    NP* b = NP::Make<float>(1,4,4) ; 
    b->fillIndexFlat(); 

    NPFold nf0 ; 
    nf0.add("a", a ); 
    nf0.add("b", b ); 
    std::cout << "nf0" << std::endl << nf0.desc()  ; 

    const char* base = "/tmp/NPFold_test/test_add_without_ext" ; 
    nf0.save(base); 

    NPFold* nf1 = NPFold::Load(base); 

    std::cout << "nf1" << std::endl << nf1->desc()  ; 
    int cf = NPFold::Compare(&nf0, nf1, true ); 
    assert( cf == 0 ); 
}


int main()
{
    //test_0();
    test_add_without_ext();

    return 0 ; 
}
