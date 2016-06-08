#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "BLog.hh"


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


int main(int argc, char** argv)
{
    BLog nl("sphere.log","info");
    nl.configure(argc, argv, "/tmp");

    //test_part();
    test_intersect();

    return 0 ; 
}




