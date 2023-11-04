#include <array>
#include "s_bb.h"

void test_AllZero()
{
    std::array<double, 6> bb_0 = {} ;  
    assert( s_bb::AllZero<double>( bb_0.data() ) ) ; 
    std::cout << "bb_0 " << s_bb::Desc(bb_0.data()) << std::endl ; 

    std::array<double, 6> bb_1 = {-0., -0., -0., 0., 0., 0. } ;  
    assert( s_bb::AllZero<double>( bb_1.data() ) ) ; 
    std::cout << "bb_1 " << s_bb::Desc(bb_1.data()) << std::endl ; 
}

void test_IncludeAABB()
{
    std::array<double, 6> bb_0 = {} ;  
    std::array<double, 6> bb_1 = {-100., -100., -100., 100., 100., 100. } ;  
    std::array<double, 6> bb_2 = {-10.,  -10.,  -10.,   10., 10.,   10. } ;  

    std::cout << "bb_0 " << s_bb::Desc(bb_0.data()) << std::endl ; 
    std::cout << "bb_1 " << s_bb::Desc(bb_1.data()) << std::endl ; 
    std::cout << "bb_2 " << s_bb::Desc(bb_2.data()) << std::endl ; 


    std::stringstream ss ; 

    s_bb::IncludeAABB( bb_0.data(), bb_1.data(), &ss ); 
    std::cout << " IncludeAABB( bb_0, bb_1 )  " << s_bb::Desc(bb_0.data()) << std::endl ; 

    s_bb::IncludeAABB( bb_0.data(), bb_2.data(), &ss  ); 
    std::cout << " IncludeAABB( bb_0, bb_2 )  " << s_bb::Desc(bb_0.data()) << std::endl ; 

    std::string str = ss.str(); 
    std::cout << str ; 
}

void test_include_point_widen()
{
    s_bb bb = {} ; 
    std::cout << " bb0 " << bb.desc() << std::endl ; 

    std::array<float,3> point = {{ 1.f, 2.f, 3.f }} ; 

    bb.include_point_widen( point.data() ); 

    std::cout << " bb1 " << bb.desc() << std::endl ; 
}

void test_include_aabb_widen()
{
    s_bb bb = {} ; 
    std::cout << " bb0 " << bb.desc() << std::endl ; 

    std::array<float,6> other_bb = {{ -100.f, -200.f, -300.f , 100.f, 200.f, 300.f }} ; 

    bb.include_aabb_widen( other_bb.data() ); 

    std::cout << " bb1 " << bb.desc() << std::endl ; 
}


void test_center_extent()
{
    s_bb bb = {} ; 
    std::cout << " bb0 " << bb.desc() << std::endl ; 

    std::array<float,6> other_bb = {{ -100.f, -200.f, -300.f , 100.f, 200.f, 300.f }} ; 

    bb.include_aabb_widen( other_bb.data() ); 

    std::cout << " bb1 " << bb.desc() << std::endl ; 


    std::array<float,  4> cef ; 
    std::array<double, 4> ced ; 

    bb.center_extent(cef.data()); 
    bb.center_extent(ced.data()); 

    std::cout << " cef " << s_bb::Desc_<float,4>(  cef.data() ) << std::endl ; 
    std::cout << " ced " << s_bb::Desc_<double,4>( ced.data() ) << std::endl ; 
}

void test_write()
{
    s_bb bb = {} ; 
    std::cout << " bb0 " << bb.desc() << std::endl ; 
    std::array<float,6> other_bb = {{ -100.f, -200.f, -300.f , 100.f, 200.f, 300.f }} ; 
    bb.include_aabb_widen( other_bb.data() ); 

    std::array<float,6> dst_bb_f ;  
    std::array<double,6> dst_bb_d ;  
    bb.write(dst_bb_f.data()); 
    bb.write(dst_bb_d.data()); 

    std::cout << "dst_bb_f : " << s_bb::Desc(dst_bb_f.data()) << std::endl ; 
    std::cout << "dst_bb_d : " << s_bb::Desc(dst_bb_d.data()) << std::endl ; 
}

int main()
{ 
    /*
    test_AllZero(); 
    test_IncludeAABB(); 
    test_include_point_widen() ;  
    test_include_aabb_widen() ;  
    test_center_extent() ;  
    */
    test_write() ;  

    return 0 ; 
}
