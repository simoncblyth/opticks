#include <cstdlib>
#include "NGLMExt.hpp"

#include "NZSphere.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"


void test_part()
{
    unsigned flags = 0 ; 
    nzsphere s = make_zsphere(0,0,0,10,-5,5, flags);
    npart p = s.part();
    p.dump("p");
}

void test_bbox()
{
    unsigned flags = 0 ; 
    nzsphere a = make_zsphere(0.f,0.f,0.f,100.f, -50.f, 50.f, flags);
    a.dump("zsph");

    nbbox bb = a.bbox();
    bb.dump("zbb");
}

void test_sdf()
{
    unsigned flags = 0 ; 
    float radius = 10.f ; 
    float zdelta_min = -radius/2.f ; 
    float zdelta_max = radius/2.f ; 

    nzsphere a = make_zsphere(0.f,0.f,0.f,radius,zdelta_min,zdelta_max, flags);

    for(float v=-2*radius ; v <= 2*radius ; v+= radius/10.f ) 
        std::cout 
             << " v   " << std::setw(10) << v 
             << " x   " << std::setw(10) << std::fixed << std::setprecision(2) << a(v,0,0)
             << " y   " << std::setw(10) << std::fixed << std::setprecision(2) << a(0,v,0)
             << " z   " << std::setw(10) << std::fixed << std::setprecision(2) << a(0,0,v)
             << " xy  " << std::setw(10) << std::fixed << std::setprecision(2) << a(v,v,0)
             << " xz  " << std::setw(10) << std::fixed << std::setprecision(2) << a(v,0,v)
             << " yz  " << std::setw(10) << std::fixed << std::setprecision(2) << a(0,v,v)
             << " xyz " << std::setw(10) << std::fixed << std::setprecision(2) << a(v,v,v)
             << std::endl 
             ; 
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_part();
    test_bbox();
    test_sdf();

    return 0 ; 
}




