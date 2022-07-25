// ./stran_test.sh 

#include "scuda.h"
#include "sqat4.h"
#include "stran.h"

void test_from_string()
{
    const char* a_str = "(-0.585,-0.805, 0.098, 0.000) (-0.809, 0.588, 0.000, 0.000) (-0.057,-0.079,-0.995, 0.000) (1022.116,1406.822,17734.953, 1.000)"  ;
    //const char* a_str = "( 0.5, 0.0, 0.0, 0.0 ) ( 0.0, 0.5, 0.0, 0.0) ( 0.0, 0.0, 0.5, 0.000) (1000.0, 1000.0,1000.0, 1.000)"  ;

    qat4* a = qat4::from_string(a_str); 

    unsigned id0[3] ;
    id0[0] = 1 ; 
    id0[1] = 10 ; 
    id0[2] = 100 ; 

    a->setIdentity( id0[0], id0[1], id0[2] );

    const qat4* i = Tran<double>::Invert( a ); 

    unsigned id1[3] ; 
    i->getIdentity( id1[0], id1[1], id1[2] ); 


    assert( id0[0] == id1[0] ); 
    assert( id0[1] == id1[1] ); 
    assert( id0[2] == id1[2] ); 


    Tran<double>* chk = Tran<double>::FromPair( a, i, 1e-3 ); 

    std::cout << chk->desc() << std::endl ; 
}


void test_Translate()
{
    glm::tvec3<double> tlate(1., 2., 3.);  
    glm::tmat4x4<double> tr = Tran<double>::Translate( tlate ); 
    std::cout << Tran<double>::Desc(tr) << std::endl ;    
}


void test_MakeRotateA2B()
{
    glm::tvec3<double> a_(1., 0., 0.);
    glm::tvec3<double> b_(1., 1., 0.) ;

    glm::tvec3<double> a = glm::normalize(a_); 
    glm::tvec3<double> b = glm::normalize(b_); 

    std::cout << " a " << Tran<double>::Desc(a) << std::endl ;    
    std::cout << " b " << Tran<double>::Desc(b) << std::endl ;    

    glm::tmat4x4<double> tr0 = Tran<double>::RotateA2B(a, b, false ); 
    std::cout << " Tran<double>::Rotate(a,b,false) " << std::endl << Tran<double>::Desc(tr0) << std::endl ;    

    glm::tmat4x4<double> tr1 = Tran<double>::RotateA2B(a, b, true ); 
    std::cout << " Tran<double>::Rotate(a,b,true) " << std::endl << Tran<double>::Desc(tr1) << std::endl ;    

    glm::tvec4<double> a4(a, 0.); 
    glm::tvec4<double> b4(b, 0.); 

    std::cout << std::setw(20) << " a4 " << Tran<double>::Desc(a4) << std::endl ;    
    std::cout << std::setw(20) << " b4 " << Tran<double>::Desc(b4) << std::endl ;    

    glm::tvec4<double> tr0_a4 = tr0 * a4 ; 
    glm::tvec4<double> tr1_a4 = tr1 * a4 ; 

    glm::tvec4<double> a4_tr0 = a4 * tr0 ; 
    glm::tvec4<double> a4_tr1 = a4 * tr1 ; 


    std::cout << std::setw(20) << " tr0_a4 " << Tran<double>::Desc(tr0_a4) << std::endl ;    
    std::cout << std::setw(20) << " tr1_a4 " << Tran<double>::Desc(tr1_a4) << " (this matches b4) " << std::endl ;    

    std::cout << std::setw(20) << " a4_tr0 " << Tran<double>::Desc(a4_tr0) << " (this matches b4) " << std::endl ;    
    std::cout << std::setw(20) << " a4_tr1 " << Tran<double>::Desc(a4_tr1) << std::endl ;    
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

    std::cout << " a " << Tran<double>::Desc(a) << std::endl ;    
    std::cout << " b " << Tran<double>::Desc(b) << std::endl ;    
    std::cout << " c " << Tran<double>::Desc(c) << std::endl ;    


    glm::tmat4x4<double> tr0 = Tran<double>::Place(a, b, c, false ); 
    std::cout << " tr0 = Tran<double>::Place(a,b,c,false) " << std::endl << Tran<double>::Desc(tr0) << std::endl ;    

    glm::tmat4x4<double> tr1 = Tran<double>::Place(a, b, c, true ); 
    std::cout << " tr1 = Tran<double>::Place(a,b,c,true) " << std::endl << Tran<double>::Desc(tr1) << std::endl ;    

    glm::tvec4<double> a4(a, 1.); 
    glm::tvec4<double> b4(b, 1.); 
    glm::tvec4<double> c4(c, 1.); 
    glm::tvec4<double> e4(b+c, 1.); 

    std::cout << std::setw(20) << " a4 " << Tran<double>::Desc(a4) << std::endl ;    
    std::cout << std::setw(20) << " b4 " << Tran<double>::Desc(b4) << std::endl ;    
    std::cout << std::setw(20) << " c4 " << Tran<double>::Desc(c4) << std::endl ;    
    std::cout << std::setw(20) << " e4 = b + c " << Tran<double>::Desc(e4) << std::endl ;    

    glm::tvec4<double> tr0_a4 = tr0 * a4 ; 
    glm::tvec4<double> tr1_a4 = tr1 * a4 ; 

    glm::tvec4<double> a4_tr0 = a4 * tr0 ; 
    glm::tvec4<double> a4_tr1 = a4 * tr1 ; 


    std::cout << std::setw(20) << " tr0_a4 " << Tran<double>::Desc(tr0_a4) << std::endl ;    
    std::cout << std::setw(20) << " tr1_a4 " << Tran<double>::Desc(tr1_a4) << "(this gives expected e4)" << std::endl ;    

    std::cout << std::setw(20) << " a4_tr0 " << Tran<double>::Desc(a4_tr0) << std::endl ;    
    std::cout << std::setw(20) << " a4_tr1 " << Tran<double>::Desc(a4_tr1) << std::endl ;    

    std::cout << std::setw(20) << " e4 = b + c  " << Tran<double>::Desc(e4) << std::endl ;    

}

void test_photon_transform()
{
    sphoton p ; 
    p.pos = {0.f, 0.f, 0.f } ; 
    p.time = 0.f  ;
    p.mom = {0.f, 0.f, 1.f } ; 
    p.pol = {0.f, 1.f, 0.f } ; 


     
 

  



}






int main()
{
    /*
    test_from_string();  
    test_Translate(); 
    test_MakeRotateA2B(); 
    test_Place(); 
    */

    test_photon_transform(); 

    return 0 ; 
}
