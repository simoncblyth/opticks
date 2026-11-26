// ~/o/sysrap/tests/squad_test.sh

#include "scuda.h"
#include "squad.h"
//#include "squad_double.h"
#include "ssys.h"


struct squad_test
{
   static int load_array();
   static int load_data();
   //static int dquad4_zero();
   static int qvals_uint3();
   static int plane();

   static int main();
};

int squad_test::main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    int rc = 0 ;
    if(ALL || 0==strcmp(TEST,"load_array"))  rc += load_array();
    if(ALL || 0==strcmp(TEST,"load_data"))   rc += load_data();
    //if(ALL || 0==strcmp(TEST,"dquad4_zero")) rc += dquad4_zero();
    if(ALL || 0==strcmp(TEST,"qvals_uint3")) rc += qvals_uint3();
    if(ALL || 0==strcmp(TEST,"plane")) rc += plane();

    return rc ;
}




int squad_test::load_array()
{
    std::array<double, 16> a ;
    for(int i=0 ; i < 16 ; i++) a[i] = double(i*100) ;

    quad4 p ;
    p.load(a) ;
    std::cout << p << std::endl ;

    return 0 ;
}

int squad_test::load_data()
{
    std::array<double, 16> a ;
    for(int i=0 ; i < 16 ; i++) a[i] = double(i*100) ;
    const double* ptr = a.data() ;

    quad4 p ;
    p.load(ptr, 16) ;
    std::cout << p << std::endl ;
    return 0 ;
}

/*
int squad_test::dquad4_zero()
{
    dquad4 dq4 ;
    dq4.zero();

    assert( sizeof(dq4) == sizeof(double)*16 );


    //std::cout << "dq4.d0.f " << dq4.q0.f  << std::endl ;
    //std::cout << "dq4      " << dq4 << std::endl ;
    //std::cout << "dq4.q0   " << dq4.q0 << std::endl ;


    return 0 ;
}
*/

int squad_test::qvals_uint3()
{
    uint3 pidxyz ;
    qvals(pidxyz, "PIDXYZ", "-1:-1:-1", -1 ) ;

    std::cout << " pidxyz" << pidxyz << "\n" ;

    return 0 ;
}

int squad_test::plane()
{
    quad q = {} ;

    unsigned num = 10 ;
    unsigned idx = 1001 ;

    q.setPlaneNum(num);
    q.setPlaneIdx(idx);

    assert( q.planeNum() == num );
    assert( q.planeIdx() == idx );

    return 0 ;
}

int main()
{
    return squad_test::main();
}
