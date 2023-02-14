// ./stran_test.sh 

#include "scuda.h"
#include "sqat4.h"
#include "stran.h"
#include "sphoton.h"


void test_from_string()
{
    const char* a_str = "(-0.585,-0.805, 0.098, 0.000) (-0.809, 0.588, 0.000, 0.000) (-0.057,-0.079,-0.995, 0.000) (1022.116,1406.822,17734.953, 1.000)"  ;
    //const char* a_str = "( 0.5, 0.0, 0.0, 0.0 ) ( 0.0, 0.5, 0.0, 0.0) ( 0.0, 0.0, 0.5, 0.000) (1000.0, 1000.0,1000.0, 1.000)"  ;

    qat4* a = qat4::from_string(a_str); 

    int id0[4] ;
    id0[0] = 1 ; 
    id0[1] = 10 ; 
    id0[2] = 100 ; 
    id0[3] = 1000 ; 

    a->setIdentity( id0[0], id0[1], id0[2], id0[3] );

    const qat4* i = Tran<double>::Invert( a ); 

    int id1[4] ; 
    i->getIdentity( id1[0], id1[1], id1[2], id1[3] ); 


    assert( id0[0] == id1[0] ); 
    assert( id0[1] == id1[1] ); 
    assert( id0[2] == id1[2] ); 
    assert( id0[3] == id1[3] ); 

    Tran<double>* chk = Tran<double>::FromPair( a, i, 1e-3 ); 

    std::cout << chk->desc() << std::endl ; 
}


void test_Translate()
{
    glm::tvec3<double> tlate(1., 2., 3.);  
    glm::tmat4x4<double> tr = stra<double>::Translate( tlate ); 
    std::cout << stra<double>::Desc(tr) << std::endl ;    
}


void test_MakeRotateA2B()
{
    glm::tvec3<double> a_(1., 0., 0.);
    glm::tvec3<double> b_(1., 1., 0.) ;

    glm::tvec3<double> a = glm::normalize(a_); 
    glm::tvec3<double> b = glm::normalize(b_); 

    std::cout << " a " << stra<double>::Desc(a) << std::endl ;    
    std::cout << " b " << stra<double>::Desc(b) << std::endl ;    

    glm::tmat4x4<double> tr0 = stra<double>::RotateA2B(a, b, false ); 
    std::cout << " stra<double>::Rotate(a,b,false) " << std::endl << stra<double>::Desc(tr0) << std::endl ;    

    glm::tmat4x4<double> tr1 = stra<double>::RotateA2B(a, b, true ); 
    std::cout << " stra<double>::Rotate(a,b,true) " << std::endl << stra<double>::Desc(tr1) << std::endl ;    

    glm::tvec4<double> a4(a, 0.); 
    glm::tvec4<double> b4(b, 0.); 

    std::cout << std::setw(20) << " a4 " << stra<double>::Desc(a4) << std::endl ;    
    std::cout << std::setw(20) << " b4 " << stra<double>::Desc(b4) << std::endl ;    

    glm::tvec4<double> tr0_a4 = tr0 * a4 ; 
    glm::tvec4<double> tr1_a4 = tr1 * a4 ; 

    glm::tvec4<double> a4_tr0 = a4 * tr0 ; 
    glm::tvec4<double> a4_tr1 = a4 * tr1 ; 


    std::cout << std::setw(20) << " tr0_a4 " << stra<double>::Desc(tr0_a4) << std::endl ;    
    std::cout << std::setw(20) << " tr1_a4 " << stra<double>::Desc(tr1_a4) << " (this matches b4) " << std::endl ;    

    std::cout << std::setw(20) << " a4_tr0 " << stra<double>::Desc(a4_tr0) << " (this matches b4) " << std::endl ;    
    std::cout << std::setw(20) << " a4_tr1 " << stra<double>::Desc(a4_tr1) << std::endl ;    
}

/**
 
Place transform is intended to rotate and then translate 
The rotation comes from RotateA2B where a and b are normalized vectors. 


             "(1,1)"
              /
           b /
            /
           /
          +-------->
               a
**/

void test_Place()
{
    glm::tvec3<double> a_(1., 0., 0.);
    glm::tvec3<double> b_(1., 1., 0.) ;
    glm::tvec3<double> c( 1., 2., 3.) ;

    glm::tvec3<double> a = glm::normalize(a_); 
    glm::tvec3<double> b = glm::normalize(b_); 

    std::cout << " a " << stra<double>::Desc(a) << std::endl ;    
    std::cout << " b " << stra<double>::Desc(b) << std::endl ;    
    std::cout << " c " << stra<double>::Desc(c) << std::endl ;    


    glm::tmat4x4<double> tr0 = stra<double>::Place(a, b, c, false ); 
    std::cout << " tr0 = stra<double>::Place(a,b,c,false) " << std::endl << stra<double>::Desc(tr0) << std::endl ;    

    glm::tmat4x4<double> tr1 = stra<double>::Place(a, b, c, true ); 
    std::cout << " tr1 = stra<double>::Place(a,b,c,true) " << std::endl << stra<double>::Desc(tr1) << std::endl ;    

    glm::tvec4<double> a4(a, 1.); 
    glm::tvec4<double> b4(b, 1.); 
    glm::tvec4<double> c4(c, 1.); 
    glm::tvec4<double> e4(b+c, 1.); 

    std::cout << std::setw(20) << " a4 " << stra<double>::Desc(a4) << std::endl ;    
    std::cout << std::setw(20) << " b4 " << stra<double>::Desc(b4) << std::endl ;    
    std::cout << std::setw(20) << " c4 " << stra<double>::Desc(c4) << std::endl ;    
    std::cout << std::setw(20) << " e4 = b + c " << stra<double>::Desc(e4) << std::endl ;    

    glm::tvec4<double> tr0_a4 = tr0 * a4 ; 
    glm::tvec4<double> tr1_a4 = tr1 * a4 ; 

    glm::tvec4<double> a4_tr0 = a4 * tr0 ; 
    glm::tvec4<double> a4_tr1 = a4 * tr1 ; 


    std::cout << std::setw(20) << " tr0_a4 " << stra<double>::Desc(tr0_a4) << std::endl ;    
    std::cout << std::setw(20) << " tr1_a4 " << stra<double>::Desc(tr1_a4) << "(this gives expected e4)" << std::endl ;    

    std::cout << std::setw(20) << " a4_tr0 " << stra<double>::Desc(a4_tr0) << std::endl ;    
    std::cout << std::setw(20) << " a4_tr1 " << stra<double>::Desc(a4_tr1) << std::endl ;    

    std::cout << std::setw(20) << " e4 = b + c  " << stra<double>::Desc(e4) << std::endl ;    
}



template<typename T>
glm::tmat4x4<T> make_translate(T tx, T ty, T tz)
{
    std::array<T, 16> aa = 
         {{1., 0., 0., 0., 
           0., 1., 0., 0., 
           0., 0., 1., 0., 
           tx, ty, tz, 1. }} ; 
    glm::tmat4x4<T> tr = stra<T>::FromData(aa.data()) ; 
    return tr ; 
}


void init( sphoton& p )
{
    p.pos = {0.f, 0.f, 0.f } ; 
    p.time = 0.f  ;
    p.mom = {0.f, 0.f, 1.f } ; 
    p.pol = {0.f, 1.f, 0.f } ; 
}


void test_photon_transform_0() 
{
    double tx = 10. ; 
    double ty = 20. ; 
    double tz = 30. ; 

    const glm::tmat4x4<double> tr = make_translate<double>(tx, ty, tz); 
    
    sphoton p0 ; 
    init(p0); 

    sphoton p1(p0); 
    p1.transform(tr);  

    std::cout << " p0.descBase " << p0.descBase() << std::endl ;  
    std::cout << " p1.descBase " << p1.descBase() << std::endl ;  

    assert( p1.pos.x = p0.pos.x + tx );  
    assert( p1.pos.y = p0.pos.y + ty );  
    assert( p1.pos.z = p0.pos.z + tz );  
}


void test_TranConvert()
{
    float tx = 10. ; 
    float ty = 20. ; 
    float tz = 30. ; 
    const glm::tmat4x4<float> tr0 = make_translate<float>(tx, ty, tz); 

    glm::tmat4x4<double> tr1 ;

    //TranConvert<double,float>(tr1, tr0);  
    TranConvert(tr1, tr0);  

    std::cout << " tr0\n" << tr0 << std::endl ; 
    std::cout << " tr1\n" << tr1 << std::endl ; 
}


int main()
{
    /*
    test_from_string();  
    test_Translate(); 
    test_MakeRotateA2B(); 
    test_Place(); 
    test_photon_transform_0(); 
    */

    test_TranConvert(); 


    return 0 ; 
}
