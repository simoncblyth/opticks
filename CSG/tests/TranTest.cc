#include <iostream>
#include "Tran.h"


void test_make()
{
    const Tran<float>* i = Tran<float>::make_translate( 0.f, 0.f, 0.f ) ; 
    const Tran<float>* t = Tran<float>::make_translate( 1.f, 2.f, 3.f ) ; 
    const Tran<float>* s = Tran<float>::make_scale(     1.f, 2.f, 3.f ) ; 
    const Tran<float>* r = Tran<float>::make_rotate(    0.f, 0.f, 1.f, 45.f ) ; 

    const Tran<float>* ts0 = Tran<float>::product( t, s, false ); 
    const Tran<float>* ts1 = Tran<float>::product( t, s, true ); 

    std::cout << "i   " << i->brief() << std::endl ; 
    std::cout << "t   " << t->brief() << std::endl ; 
    std::cout << "s   " << s->brief() << std::endl ; 
    std::cout << "r   " << r->brief() << std::endl ; 
    std::cout << "ts0 " << ts0->brief() << std::endl ; 
    std::cout << "ts1 " << ts1->brief() << std::endl ; 
}


void test_ctor()
{
    const Tran<float>* s = Tran<float>::make_scale( 1.f, 2.f, 3.f ) ; 
    const Tran<float>* s2 = new Tran<float>( s->tdata(), s->vdata() ) ; 

    std::cout << "s    " << s->brief() << std::endl ; 
    std::cout << "s2   " << s2->brief() << std::endl ; 
}


int main()
{
    //test_make(); 
    test_ctor(); 
    return 0 ; 
}
