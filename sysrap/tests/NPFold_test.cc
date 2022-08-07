// ./NPFold_test.sh

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

NPFold* make_NPFold(const char* opt)
{
    NPFold* nf = new NPFold ; 
    if( strchr( opt, 'a') )
    {
        NP* a = NP::Make<float>(1,4,4) ; 
        a->fillIndexFlat(); 
        nf->add("a", a); 
    }
    if( strchr( opt, 'b') )
    {
        NP* b = NP::Make<float>(2,4,4) ; 
        b->fillIndexFlat(); 
        nf->add("b", b); 
    }
    if( strchr( opt, 'c') )
    {
        NP* c = NP::Make<float>(3,4,4) ; 
        c->fillIndexFlat(); 
        nf->add("c", c);
    } 
    return nf ; 
}

/**

    nf0
       a.npy
       b.npy
       c.npy
       nf1 
           a.npy
           b.npy 
           nf2
               b.npy
               c.npy 

**/

NPFold* make_compound()
{
    NPFold* nf0 = make_NPFold("abc"); 

        NPFold* nf1 = make_NPFold("ab"); 
        nf0->add_subfold( "nf1", nf1 ); 

            NPFold* nf2 = make_NPFold("bc"); 
            NPFold* nf3 = make_NPFold("ac"); 
            NPFold* nf4 = make_NPFold("b"); 
            NPFold* nf5 = make_NPFold("a"); 

            nf1->add_subfold("nf2", nf2 ); 
            nf1->add_subfold("nf3", nf3 ); 
            nf1->add_subfold("nf4", nf4 ); 
            {
                NPFold* nf6 = make_NPFold("ac") ;
                nf4->add_subfold("nf6", nf6 ); 
            }
            nf1->add_subfold("nf5", nf5 ); 

    return nf0 ; 
}


void test_subfold_save_load()
{
    const char* base = "/tmp/NPFold_test/test_subfold" ; 

    NPFold* nf0 = make_compound(); 
    std::cout << "nf0" << std::endl << nf0->desc()  ; 
    nf0->save(base); 

    NPFold* nfl = NPFold::Load(base); 
    std::cout << "nfl" << std::endl << nfl->desc()  ; 

    NPFold* nfl_nf1 = nfl->get_subfold("nf1"); 
    std::cout << "nfl_nf1" << std::endl << nfl_nf1->desc()  ; 
    NPFold* nfl_nf1_nf2 = nfl_nf1->get_subfold("nf2"); 

    std::cout << "nfl_nf1_nf2" << std::endl << nfl_nf1_nf2->desc()  ; 
}

void test_desc_subfold()
{
    NPFold* nf0 = make_compound(); 
    std::cout << "[nf0.desc_subfold\n " << nf0->desc_subfold("nf0") << "]" << std::endl ; 
}
void test_find_subfold_0()
{
    NPFold* nf0 = make_compound(); 
    NPFold* nf4 = nf0->find_subfold("nf1/nf4"); 
    std::cout << "[nf0->find_subfold('nf1/nf4')\n " << ( nf4 ? nf4->desc() : "-" ) << "]" << std::endl ; 
}
void test_find_subfold_1()
{
    NPFold* nf0 = make_compound(); 
    NPFold* nf6 = nf0->find_subfold("nf1/nf4/nf6"); 
    std::cout << "[nf0->find_subfold('nf1/nf4/nf6')\n " << ( nf6 ? nf6->desc() : "-" ) << "]" << std::endl ; 
}



int main()
{
    /*
    test_0();
    test_add_without_ext();
    test_subfold_save_load(); 
    test_desc_subfold(); 
    test_find_subfold_0(); 
    */
    test_find_subfold_1(); 

    return 0 ; 
}
