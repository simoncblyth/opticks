// name=qgs_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I../../sysrap -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name
/**
qgs_test.cc
==============

Using union to call things by different names::

    What's in a name? That which we call a rose
    By any other name would smell as sweet;

* when loading and saving in and out of buffers the quad6 lingo is most convenient.
* when using the genstep the more specific naming from 
  the same storage struct is more convenient. 

**/
#include "vector_types.h"
#include "scuda.h"
#include "squad.h"

//#include "qgs.h"

struct TO
{
    unsigned gencode ; 
    unsigned q0_y ; 
    unsigned q0_z ; 
    unsigned q0_w ;

    quad q1 ; 
    quad q2 ; 
    quad q3 ; 
    quad q4 ;
 
    float q5x ; 
    float q5y ; 
    float q5z ; 
    float q5w ; 
}; 

struct QT
{
    union 
    {
        quad6 q ; 
        TO    t ; 
    }; 
}; 


int main(int argc, char** argv)
{
    quad6 gs ; 
    gs.zero(); 

    gs.q0.u = make_uint4( 1u, 2u, 3u, 4u ); 
    gs.q5.f = make_float4( 1.f, 2.f, 3.f, 4.f ); 

    QT qt ; 
    qt.q = gs ; 
   
    printf("// qt.t.gencode  %d \n", qt.t.gencode  ); 
    printf("// qt.t.q5w      %10.4f \n", qt.t.q5w ); 

    return 0 ; 
}
