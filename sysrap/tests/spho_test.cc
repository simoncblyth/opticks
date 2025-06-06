// ./spho_test.sh


#include <iostream>
#include "spho.h"

void test_gen()
{
     spho p = {} ;
     p.id = 101 ;
     std::cout << p.desc() << std::endl ;

     for(unsigned i=0 ; i < 256 ; i++)
     {
         if( i == 10 ) p.uc4.w = 214 ;

         p = p.make_nextgen();
         std::cout << p.desc() << std::endl ;
     }



     p.set_gen(1);
     std::cout
         << p.desc() << " "
         << " p.set_gen(1) "
         << std::endl
         ;

     p.set_gen(0);
     p.set_flg(1);

     std::cout
         << p.desc() << " "
         << "p.set_gen(0), p.set_flg(1) "
         << std::endl
         ;


     p.set_gen(1);
     p.set_flg(1);
     std::cout
         << p.desc() << " "
         << "p.set_gen(1), p.set_flg(1) "
         << std::endl
         ;
}


void test_u4c()
{
     spho p = {} ;

     p.uc4.x = 1 ;
     p.uc4.y = 1 ;
     p.uc4.z = 1 ;
     p.uc4.w = 1 ;
     std::cout
         << p.desc() << " "
         << "p.uc4.x = 1, p.uc4.y = 1, p.uc4.z = 1, p.uc4.w = 1 "
         << std::endl
         ;



     p.uc4.x = 0xff ;
     p.uc4.y = 0xff ;
     p.uc4.z = 0xff ;
     p.uc4.w = 0xff ;
     std::cout
         << p.desc() << " "
         << "p.uc4.x = 0xff, p.uc4.y = 0xff, p.uc4.z = 0xff, p.uc4.w = 0xff "
         << std::endl
         ;

     p.uc4.x = 'A' ;
     p.uc4.y = 'a' ;
     p.uc4.z = '?' ;
     p.uc4.w = '_' ;
     std::cout
         << p.desc() << " "
         << "p.uc4.x = 'A' p.uc4.y = 'a' p.uc4.z = '?'  p.uc4.w = '_'  "
         << std::endl
         ;


     p.uc4.x = 'Z' ;
     p.uc4.y = 'z' ;
     p.uc4.z = ' ' ;
     p.uc4.w = '_' ;
     std::cout
         << p.desc() << " "
         << "p.uc4.x = 'Z' p.uc4.y = 'z' p.uc4.z = ' '  p.uc4.w = '_'  "
         << std::endl
         ;
}

void test_serialize_load()
{
    spho p = { 1, 2, 3, {'a', 'b', 'c', 'd' } };

    std::cout << "p " << p << std::endl;

    std::array<int,4> a ;
    p.serialize(a) ;

    spho q = {} ;
    q.load(a) ;

    std::cout << "q " << q << std::endl ;

}

/**

::

    In [6]: np.array( [1684234849], dtype=np.uint32 ).view("|S4")
    Out[6]: array([b'abcd'], dtype='|S4')

**/

void test_uc4packed()
{
    spho p = { 1, 2, 3, {'a', 'b', 'c', 'd' } };

    unsigned u4pk = p.uc4packed() ;
    std::cout << " u4pk " << u4pk << std::endl ;
}


void test_inplace()
{
    // inplace modify the p object via another object with only int* data connection between them

    spho p = { 1, 2, 3, {'a', 'b', 'c', 'd' } };
    std::cout << "p0 " << p << std::endl ;
    int* d = p.data(); // d points into p


    spho& q = reinterpret_cast<spho&>(*d);
    std::cout << "q  " << q << std::endl ;

    // changing q also changes p
    q.uc4.x = 'A' ;
    q.uc4.y = 'B' ;
    q.uc4.z = 'C' ;
    q.uc4.w = 'D' ;
    q.gs = 10 ;
    q.ix = 20 ;
    q.id = 30 ;

    std::cout << "p1 " << p << std::endl ;
    std::cout << "q  " << q << "(q and p share same data)" << std::endl ;
}


int main()
{
     /*
     test_gen();
     test_uc4();
     test_serialize_load();
     test_uc4packed();
     */
     test_inplace();

     return 0 ;
}
