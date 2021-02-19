// TEST=NGLM2Test om-t

#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include <glm/gtx/matrix_interpolation.hpp>
#include "OPTICKS_LOG.hh"


/*

PBRT 2nd Edition, p97

Next we’d like to extract the pure rotation component of M. We’ll use a
technique called polar decomposition to do this. It can be shown that the polar
decomposition of a matrix M into rotation R and scale S can be computed by
successively averaging M to its inverse transpose

... if M is a pure rotation, then averaging it with its inverse transpose 
will leave it unchanged, since its inverse is equal to its transpose.

*/


void test_polar_decomposition_pluck_scale()
{
    LOG(info) ;
 
    glm::vec3 tla(10,0,10);
    glm::vec4 rot(1,1,1,45);   // 45 degrees about some axes
    glm::vec3 sca(1,2,3);

    glm::mat4 s   = nglmext::make_transform("s", tla, rot, sca);
    glm::mat4 r   = nglmext::make_transform("r", tla, rot, sca);
    glm::mat4 t   = nglmext::make_transform("t", tla, rot, sca);
    glm::mat4 trs = nglmext::make_transform("trs", tla, rot, sca);

    ndeco d ;
    bool verbose = false ; 
    nglmext::polar_decomposition( trs, d, verbose );

    std::cout << gpresent( "trs", trs ) << std::endl ;
    std::cout << gpresent( "d.trs", d.trs ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "s", s ) << std::endl ;
    std::cout << gpresent( "d.s", d.s ) << std::endl ;
    std::cout << gpresent( "d.is", d.is ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "r", r ) << std::endl ;
    std::cout << gpresent( "d.r", d.r ) << std::endl ;
    std::cout << gpresent( "d.ir", d.ir ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "t", t ) << std::endl ;
    std::cout << gpresent( "d.t", d.t ) << std::endl ;
    std::cout << gpresent( "d.it", d.it ) << std::endl ;
    std::cout << std::endl; 


    glm::vec3 dsca = nglmext::pluck_scale( d ); 
    std::cout << gpresent( "dsca", dsca ) << std::endl ;

    bool has_scale = nglmext::has_scale( dsca ); 
    std::cout << " has_scale " << has_scale << std::endl ; 
}


void test_polar_decomposition_trs()
{
    LOG(info) << "test_polar_decomposition_trs" ; 

    glm::vec3 tla(0,0,100);
    glm::vec4 rot(0,0,1,45);
    glm::vec3 sca(2.f);

    glm::mat4 s   = nglmext::make_transform("s", tla, rot, sca);
    glm::mat4 r   = nglmext::make_transform("r", tla, rot, sca);
    glm::mat4 t   = nglmext::make_transform("t", tla, rot, sca);
    glm::mat4 trs = nglmext::make_transform("trs", tla, rot, sca);

    ndeco d ;
    bool verbose = false ; 
    nglmext::polar_decomposition( trs, d, verbose );

    std::cout << gpresent( "trs", trs ) << std::endl ;
    std::cout << gpresent( "d.trs", d.trs ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "s", s ) << std::endl ;
    std::cout << gpresent( "d.s", d.s ) << std::endl ;
    std::cout << gpresent( "d.is", d.is ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "r", r ) << std::endl ;
    std::cout << gpresent( "d.r", d.r ) << std::endl ;
    std::cout << gpresent( "d.ir", d.ir ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "t", t ) << std::endl ;
    std::cout << gpresent( "d.t", d.t ) << std::endl ;
    std::cout << gpresent( "d.it", d.it ) << std::endl ;
    std::cout << std::endl; 

    glm::mat4 i_trs = glm::inverse(d.trs);

    std::cout << gpresent( "i_trs", i_trs ) << std::endl ;
    std::cout << gpresent( "d.isirit", d.isirit ) << std::endl ;

    glm::mat4 i_trs_x_trs = i_trs * trs ;  
    std::cout << gpresent( "i_trs_x_trs", i_trs_x_trs ) << std::endl ;

    glm::mat4 isirit_x_trs = d.isirit * trs ;  
    std::cout << gpresent( "isirit_x_trs", isirit_x_trs ) << std::endl ;
}



template<typename T>
void test_polar_decomposition_trs_()
{
    LOG(info) << "test_polar_decomposition_trs_" ; 

    glm::tvec3<T> tla(0.,0.,100.);
    glm::tvec4<T> rot(0.,0.,1.,45.);
    glm::tvec3<T> sca(2.);

    glm::tmat4x4<T> s   = nglmext::make_transform_("s", tla, rot, sca);
    glm::tmat4x4<T> r   = nglmext::make_transform_("r", tla, rot, sca);
    glm::tmat4x4<T> t   = nglmext::make_transform_("t", tla, rot, sca);
    glm::tmat4x4<T> trs = nglmext::make_transform_("trs", tla, rot, sca);

    std::cout << gpresent__( "s", s ) << std::endl ;

    ndeco_<T> d ;
    bool verbose = false ; 
    nglmext::polar_decomposition_( trs, d, verbose );

    std::cout << gpresent__( "trs", trs ) << std::endl ;
    std::cout << gpresent__( "d.trs", d.trs ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent__( "s", s ) << std::endl ;
    std::cout << gpresent__( "d.s", d.s ) << std::endl ;
    std::cout << gpresent__( "d.is", d.is ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent__( "r", r ) << std::endl ;
    std::cout << gpresent__( "d.r", d.r ) << std::endl ;
    std::cout << gpresent__( "d.ir", d.ir ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent__( "t", t ) << std::endl ;
    std::cout << gpresent__( "d.t", d.t ) << std::endl ;
    std::cout << gpresent__( "d.it", d.it ) << std::endl ;
    std::cout << std::endl; 

    glm::tmat4x4<T> i_trs = glm::inverse(d.trs);

    std::cout << gpresent__( "i_trs", i_trs ) << std::endl ;
    std::cout << gpresent__( "d.isirit", d.isirit ) << std::endl ;

    glm::tmat4x4<T> i_trs_x_trs = i_trs * trs ;  
    std::cout << gpresent__( "i_trs_x_trs", i_trs_x_trs ) << std::endl ;

    glm::tmat4x4<T> isirit_x_trs = d.isirit * trs ;  
    std::cout << gpresent__( "isirit_x_trs", isirit_x_trs ) << std::endl ;
}

template void test_polar_decomposition_trs_<float>() ; 
template void test_polar_decomposition_trs_<double>() ; 


template<typename T>
void test_make_transform_()
{
    const char* s = "0.368736 0.5 0.783603 0 0.21289 -0.866025 0.452414 0 0.904827 1.58207e-15 -0.425779 0 -0.000488281 -1.27856e-11 18980.7 1" ; 
    std::cout << " s " << s << std::endl ;  
    glm::tmat4x4<T> t = nglmext::make_transform_<T>(s); 
    std::cout << gpresent__("t", t ) << std::endl ; 
}

template void test_make_transform_<float>() ;
template void test_make_transform_<double>() ;

template<typename T>
void test_invert_trs_()
{
    const char* s = "0.368736 0.5 0.783603 0 0.21289 -0.866025 0.452414 0 0.904827 1.58207e-15 -0.425779 0 -0.000488281 -1.27856e-11 18980.7 1" ; 
    glm::tmat4x4<T> t = nglmext::make_transform_<T>(s); 
    std::cout << gpresent__("t", t ) << std::endl ; 

    bool match = true ; 
    glm::tmat4x4<T> v = nglmext::invert_trs_<T>( t, match ); 
    std::cout << " match " << match << std::endl ; 

    std::cout << gpresent__("v", v ) << std::endl ; 
}

template void test_invert_trs_<float>() ;
template void test_invert_trs_<double>() ;


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    
    //test_polar_decomposition_trs();
    //test_polar_decomposition_trs_<float>();
    //test_polar_decomposition_trs_<double>();

    //test_make_transform_<float>(); 
    //test_make_transform_<double>(); 

    test_invert_trs_<float>(); 
    test_invert_trs_<double>(); 

    return 0 ; 
}


