#include <iostream>
#include <csignal>

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"

#include "SPath.hh"
#include "NP.hh"


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
    std::cout  << *t << std::endl ; 

    Tran<double>* trd = Tran<double>::ConvertToTran(t); 
    qat4* trd_t = Tran<double>::ConvertFrom(trd->t); 
    qat4* trd_v = Tran<double>::ConvertFrom(trd->v); 
    qat4* trd_i = Tran<double>::ConvertFrom(trd->i); 

    int rc = qat4::compare( *t, *trd_t, 1e-7 ) ; 
    assert( rc == 0 ); 
    if(rc!=0) std::raise(SIGINT); 


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


template<typename T>
void test_write(const char* name)
{
    const char* fold = SPath::Resolve("$TMP/sysrap/stranTest/test_write", DIRPATH); 

    const Tran<T>* tr = Tran<T>::make_rotate(    0.f, 0.f, 1.f, 45.f ) ; 

    NP* a = NP::Make<T>(3, 4, 4 ); 
    tr->write( a->values<T>() ) ; 

    a->dump(); 
    a->save(fold, name) ; 
}

template<typename T>
void test_apply()
{
    const Tran<T>* tr = Tran<T>::make_translate(0., 0., 100.) ; 

    T pos[8] ; 
    pos[0] = 0. ; 
    pos[1] = 0. ; 
    pos[2] = 0. ; 
    pos[3] = 0. ;
    pos[4] = 0. ; 
    pos[5] = 0. ; 
    pos[6] = 0. ; 
    pos[7] = 0. ;

    unsigned count = 2 ; 
    unsigned stride = 4 ; 
    unsigned offset = 0 ; 

    T w = 1. ; 
  
    tr->apply( &pos[0], w, count, stride, offset ); 

    for(int i=0 ; i < 8 ; i++) std::cout << std::setw(1) << i << " : " << pos[i] << std::endl ; 
}

template<typename T>
void test_PhotonTransform()
{
     double data[16] = { 
           1., 0., 0.,   0., 
           0., 1., 0.,   0., 
           0., 0.,-1.,   0., 
           0., 0., 100., 1. } ; 

     const Tran<T>* t = Tran<T>::ConvertFromData(&data[0]) ; 
     //const Tran<T>* t = Tran<T>::make_translate(0., 0., 100.) ; 


     const char* name = "RandomDisc100_f8.npy" ; 
     const char* path = SPath::Resolve("$HOME/.opticks/InputPhotons" , name, NOOP ); 

     NP* a = NP::Load(path); 
     NP* b0 = Tran<T>::PhotonTransform(a, false, t); 
     NP* b1 = Tran<T>::PhotonTransform(a, true, t); 
 

     const char* FOLD = SPath::Resolve("$TMP/stranTest", DIRPATH ); 
     std::cout << " FOLD " << FOLD << std::endl ; 

     a->save(FOLD,  "a.npy"); 
     b0->save(FOLD, "b0.npy"); 
     b1->save(FOLD, "b1.npy"); 
     t->save(FOLD,  "t.npy"); 
}






int main()
{
    /*
    test_make(); 
    test_ctor(); 
    test_Convert();
    test_write<float>("f.npy"); 
    test_write<double>("d.npy"); 
    test_apply<double>(); 
    */
    test_PhotonTransform<double>(); 


    return 0 ; 
}
// om- ; TEST=stranTest om-t


