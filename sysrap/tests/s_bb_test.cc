#include <array>
#include "ssys.h"
#include "s_bb.h"

struct s_bb_test
{
    static int Main();

    static int AllZero();
    static int Degenerate();
    static int IncludeAABB();
    static int include_point_widen();
    static int include_aabb_widen();
    static int center_extent();
    static int write();
};


int s_bb_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST,"ALL") == 0 ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"AllZero")) rc += AllZero();
    if(ALL||0==strcmp(TEST,"Degenerate")) rc += Degenerate();
    if(ALL||0==strcmp(TEST,"IncludeAABB")) rc += IncludeAABB();
    if(ALL||0==strcmp(TEST,"include_point_widen")) rc += include_point_widen();
    if(ALL||0==strcmp(TEST,"include_aabb_widen")) rc += include_aabb_widen();
    if(ALL||0==strcmp(TEST,"center_extent")) rc += center_extent();
    if(ALL||0==strcmp(TEST,"write")) rc += write();

    return rc ;
}


int s_bb_test::AllZero()
{
    std::array<double, 6> bb_0 = {} ;
    assert( s_bb::AllZero<double>( bb_0.data() ) ) ;
    std::cout << "bb_0 " << s_bb::Desc(bb_0.data()) << std::endl ;

    std::array<double, 6> bb_1 = {-0., -0., -0., 0., 0., 0. } ;
    assert( s_bb::AllZero<double>( bb_1.data() ) ) ;
    std::cout << "bb_1 " << s_bb::Desc(bb_1.data()) << std::endl ;

    return 0 ;
}

int s_bb_test::Degenerate()
{
    std::array<double, 6> bb_0 = {} ;
    assert( s_bb::Degenerate<double>( bb_0.data() ) ) ;
    std::cout << "bb_0 " << s_bb::Desc(bb_0.data()) << std::endl ;

    std::array<double, 6> bb_1 = {-0., -0., -0., 0., 0., 0. } ;
    assert( s_bb::Degenerate<double>( bb_1.data() ) ) ;
    std::cout << "bb_1 " << s_bb::Desc(bb_1.data()) << std::endl ;

    std::array<double, 6> bb_2 = {-1000., -2000., -3000., -1000., -2000., -3000. } ;
    assert( s_bb::Degenerate<double>( bb_2.data() ) ) ;
    std::cout << "bb_2 " << s_bb::Desc(bb_2.data()) << std::endl ;

    std::array<double, 6> bb_3 = {-1000., -2000., -3000.,  1000., 2000., 3000. } ;
    assert( s_bb::Degenerate<double>( bb_3.data() ) == false ) ;
    std::cout << "bb_3 " << s_bb::Desc(bb_3.data()) << std::endl ;


    return 0 ;
}



int s_bb_test::IncludeAABB()
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
    return 0 ;
}

int s_bb_test::include_point_widen()
{
    s_bb bb = {} ;
    std::cout << " bb0 " << bb.desc() << std::endl ;

    std::array<float,3> point = {{ 1.f, 2.f, 3.f }} ;

    bb.include_point_widen( point.data() );

    std::cout << " bb1 " << bb.desc() << std::endl ;
    return 0 ;
}

int s_bb_test::include_aabb_widen()
{
    s_bb bb = {} ;
    std::cout << " bb0 " << bb.desc() << std::endl ;

    std::array<float,6> other_bb = {{ -100.f, -200.f, -300.f , 100.f, 200.f, 300.f }} ;

    bb.include_aabb_widen( other_bb.data() );

    std::cout << " bb1 " << bb.desc() << std::endl ;
    return 0 ;
}


int s_bb_test::center_extent()
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
    return 0 ;
}

int s_bb_test::write()
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
    return 0 ;
}




int main()
{
    return s_bb_test::Main() ;
}
