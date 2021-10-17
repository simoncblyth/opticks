#include <iostream>

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"

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

void test_Convert()
{
    const char* t_str = "(-0.585,-0.805, 0.098, 0.000) (-0.809, 0.588, 0.000, 0.000) (-0.057,-0.079,-0.995, 0.000) (1022.116,1406.822,17734.953, 1.000)"  ;
    qat4* t = qat4::from_string(t_str); 

    Tran<double>* trd = Tran<double>::ConvertToTran(t); 
    qat4* trd_t = Tran<double>::ConvertFrom(trd->t); 
    qat4* trd_v = Tran<double>::ConvertFrom(trd->v); 
    qat4* trd_i = Tran<double>::ConvertFrom(trd->i); 

    std::cout  << *trd << std::endl ; 
    std::cout  << "trd_t " << *trd_t << std::endl ; 
    std::cout  << "trd_v " << *trd_v << std::endl ; 
    std::cout  << "trd_i " << *trd_i << std::endl ; 

    Tran<float>* trf = Tran<float>::ConvertToTran(t); 
    qat4* trf_t = Tran<double>::ConvertFrom(trf->t); 
    qat4* trf_v = Tran<double>::ConvertFrom(trf->v); 
    qat4* trf_i = Tran<double>::ConvertFrom(trf->i); 

    std::cout  << *trf << std::endl ; 
    std::cout  << "trf_t " << *trf_t << std::endl ; 
    std::cout  << "trf_v " << *trf_v << std::endl ; 
    std::cout  << "trf_i " << *trf_i << std::endl ; 
}

int main()
{
    //test_make(); 
    //test_ctor(); 
    test_Convert();
    return 0 ; 
}
