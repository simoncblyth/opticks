#include "NQuad.hpp"

int main()
{
    nquad qu, qf, qi, qv ;

    qu.u = {1,1,1,1}  ;
    qf.f = {1,1,1,1} ;
    qi.i = {1,1,1,1} ;

    qu.dump("qu");
    qf.dump("qf");
    qi.dump("qi");


    int v1 = 1065353216 ; // integer behind floating point 1.

    qv.i = {v1+0,v1-1,v1+1,v1+2} ;
    qv.dump("qv");

    float ulp[4];

    ulp[0] = qv.f.x - qv.f.x ; 
    ulp[1] = qv.f.y - qv.f.x ; 
    ulp[2] = qv.f.z - qv.f.x ; 
    ulp[3] = qv.f.w - qv.f.x ; 

    printf("%.10e \n", ulp[0] );
    printf("%.10e \n", ulp[1] );
    printf("%.10e \n", ulp[2] );
    printf("%.10e \n", ulp[3] );


    return 0 ;
}
