#include <cstdlib>
#include "NGLMExt.hpp"

#include "NGenerator.hpp"
#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "NBBox.hpp"

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

    nunion u = make_nunion( &a, &b );
    nintersection i = make_nintersection( &a, &b ); 
    ndifference d1 = make_ndifference( &a, &b ); 
    ndifference d2 = make_ndifference( &b, &a ); 
    nunion u2 = make_nunion( &d1, &d2 );

    typedef std::vector<nnode*> VN ;

    VN nodes ; 
    nodes.push_back( (nnode*)&a );
    nodes.push_back( (nnode*)&b );
    nodes.push_back( (nnode*)&u );
    nodes.push_back( (nnode*)&i );
    nodes.push_back( (nnode*)&d1 );
    nodes.push_back( (nnode*)&d2 );
    nodes.push_back( (nnode*)&u2 );

    for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
    {
        nnode* n = *it ; 
        OpticksCSG_t type = n->type ; 
        const char* name = n->csgname();
        std::cout 
                  << " type: " << std::setw(3) << type 
                  << " name: " << ( name ? name : "-" ) 
                  << " sdf(0,0,0): " << std::setw(10) << std::fixed << std::setprecision(2) << (*n)(0,0,0)
                  << std::endl 
                  ; 

    }

    float x = 0.f ; 
    float y = 0.f ; 
    float z = 0.f ; 

    for(int iz=-200 ; iz <= 200 ; iz+= 10, z=iz ) 
    {
        std::cout << " z  " << std::setw(10) << z 
             << " a  " << std::setw(10) << std::fixed << std::setprecision(2) << a(x,y,z) 
             << " b  " << std::setw(10) << std::fixed << std::setprecision(2) << b(x,y,z) 
             << " u  " << std::setw(10) << std::fixed << std::setprecision(2) << u(x,y,z) 
             << " i  " << std::setw(10) << std::fixed << std::setprecision(2) << i(x,y,z) 
             << " d1 " << std::setw(10) << std::fixed << std::setprecision(2) << d1(x,y,z) 
             << " d2 " << std::setw(10) << std::fixed << std::setprecision(2) << d2(x,y,z) 
             << " u2 " << std::setw(10) << std::fixed << std::setprecision(2) << u2(x,y,z) 
             << std::endl 
             ; 

        std::cout << " z  " << std::setw(10) << z  ;
        for(VN::const_iterator it=nodes.begin() ; it != nodes.end() ; it++)
        {
             nnode* n = *it ; 
             const char* name = n->csgname();
             printf(" %.3s %10.2f", ( name ? name : "-" ), (*n)(x,y,z) ); 
        }
        std::cout << std::endl ; 
    }
}


void test_bbox()
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    a.dump("sph");

    nbbox bb = a.bbox();
    bb.dump("bb");
}

void test_bbox_u()
{
    nsphere a = make_nsphere(0.f,0.f,-50.f,100.f);
    nsphere b = make_nsphere(0.f,0.f, 50.f,100.f);
    nunion  u = make_nunion( &a, &b );

    a.dump("(a) sph");
    b.dump("(b) sph");
    u.dump("(u) union(a,b)");

    nbbox a_bb = a.bbox();
    a_bb.dump("(a) bb");

    nbbox b_bb = b.bbox();
    b_bb.dump("(b) bb");

    nbbox u_bb = u.bbox();
    u_bb.dump("(u) bb");
}


void test_gtransform()
{
    nbbox bb ; 
    bb.min = {-200.f, -200.f, -200.f };
    bb.max = { 200.f,  200.f,  200.f };

    NGenerator gen(bb);

    bool verbose = !!getenv("VERBOSE") ; 
    glm::vec3 tlate ;

    for(int i=0 ; i < 100 ; i++)
    {
        gen(tlate); 

        glm::mat4 tr = glm::translate(glm::mat4(1.0f), tlate );
        glm::mat4 irit = nglmext::invert_tr(tr);
        nmat4pair mp(tr, irit);

        if(verbose)
        std::cout << " gtransform " << mp << std::endl ; 

        nsphere a = make_nsphere(0.f,0.f,0.f,100.f);      
        // untouched sphere at origin

        nsphere b = make_nsphere(0.f,0.f,0.f,100.f);      
        b.gtransform = &mp ; 
        // translated sphere via gtransform

        nsphere c = make_nsphere( tlate.x, tlate.y, tlate.z,100.f);  
        // manually positioned sphere at tlate-d position 


        float x = 0 ; 
        float y = 0 ; 
        float z = 0 ; 

        for(int iz=-200 ; iz <= 200 ; iz+= 10 ) 
        {
           z = iz ;  
           float a_ = a(x,y,z) ;
           float b_ = b(x,y,z) ;
           float c_ = c(x,y,z) ;
      
           if(verbose) 
           std::cout 
                 << " z " << std::setw(10) << z 
                 << " a_ " << std::setw(10) << std::fixed << std::setprecision(2) << a_
                 << " b_ " << std::setw(10) << std::fixed << std::setprecision(2) << b_
                 << " c_ " << std::setw(10) << std::fixed << std::setprecision(2) << c_
                 << std::endl 
                 ; 

           assert( b_ == c_ );

        }
    }
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

/*
    test_part();
    test_intersect();

    test_sdf();
    test_csgsdf();

    test_bbox();
    test_bbox_u();
*/

    test_gtransform();

    return 0 ; 
}




