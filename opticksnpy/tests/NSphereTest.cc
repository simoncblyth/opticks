#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"

#include "PLOG.hh"


void test_part()
{
    nsphere s = make_nsphere(0,0,3,10);
    npart p = s.part();
    p.dump("p");
}

void test_intersect()
{
    nsphere s1 = make_nsphere(0,0,3,10);
    nsphere s2 = make_nsphere(0,0,1,10);

    ndisc d12 = nsphere::intersect(s1,s2) ;
    d12.dump("d12");


    npart s1l = s1.zlhs(d12);
    s1l.dump("s1l");

    npart s1r = s1.zrhs(d12);
    s1r.dump("s1r");
}



void test_sdf()
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);

    float x = 0.f ; 
    float y = 0.f ; 
    float z = 0.f ; 

    for(int iz=-200 ; iz <= 200 ; iz+= 10, z=iz ) 
        std::cout 
             << " z " << std::setw(10) << z 
             << " a " << std::setw(10) << std::fixed << std::setprecision(2) << a(x,y,z)
             << std::endl 
             ; 
}


void test_csgsdf()
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_nsphere(0.f,0.f, 50.f,100.f);

    nunion u ; 
    u.left = &a  ;
    u.right = &b ;

    nintersection i ; 
    i.left = &a  ;
    i.right = &b ;

    ndifference d1 ; 
    d1.left = &a  ;
    d1.right = &b ;

    ndifference d2 ; 
    d2.left = &b  ;
    d2.right = &a ;

    nunion u2 ; 
    u2.left = &d1 ;
    u2.right = &d2 ;


    float x = 0.f ; 
    float y = 0.f ; 
    float z = 0.f ; 

    for(int iz=-200 ; iz <= 200 ; iz+= 10, z=iz ) 
        std::cout 
             << " z  " << std::setw(10) << z 
             << " a  " << std::setw(10) << std::fixed << std::setprecision(2) << a(x,y,z) 
             << " b  " << std::setw(10) << std::fixed << std::setprecision(2) << b(x,y,z) 
             << " u  " << std::setw(10) << std::fixed << std::setprecision(2) << u(x,y,z) 
             << " i  " << std::setw(10) << std::fixed << std::setprecision(2) << i(x,y,z) 
             << " d1 " << std::setw(10) << std::fixed << std::setprecision(2) << d1(x,y,z) 
             << " d2 " << std::setw(10) << std::fixed << std::setprecision(2) << d2(x,y,z) 
             << " u2 " << std::setw(10) << std::fixed << std::setprecision(2) << u2(x,y,z) 
             << std::endl 
             ; 

}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_part();
    test_intersect();

    test_sdf();
    test_csgsdf();

    return 0 ; 
}




