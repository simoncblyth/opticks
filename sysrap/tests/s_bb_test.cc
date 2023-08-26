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

    s_bb::IncludeAABB( bb_0.data(), bb_1.data() ); 
    std::cout << " IncludeAABB( bb_0, bb_1 )  " << s_bb::Desc(bb_0.data()) << std::endl ; 

    s_bb::IncludeAABB( bb_0.data(), bb_2.data() ); 
    std::cout << " IncludeAABB( bb_0, bb_2 )  " << s_bb::Desc(bb_0.data()) << std::endl ; 
}


int main()
{ 
    test_AllZero(); 
    test_IncludeAABB(); 

    return 0 ; 
}
