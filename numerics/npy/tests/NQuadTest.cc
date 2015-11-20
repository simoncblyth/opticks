#include "NQuad.hpp"
#include "GLMPrint.hpp"

void p(const nquad& q, const char* msg)
{
    printf("%s\n", msg);
    print(q.f, "q.f");
    print(q.u, "q.u");
    print(q.i, "q.i");
}


int main()
{
    nquad qu(glm::uvec4(1,1,1,1)) ;
    nquad qf(glm::vec4(1,1,1,1)) ;
    nquad qi(glm::ivec4(1,1,1,1)) ;

    p(qu, "qu(1,1,1,1)" );
    p(qf, "qf(1,1,1,1)" );
    p(qi, "qi(1,1,1,1)" );

    //int v10 = 1092616192 ; // integer behind floating point 10.
    int v1 = 1065353216 ; // integer behind floating point 1.

    nquad qv(glm::ivec4(v1+0,v1-1,v1+1,v1+2));
    p(qv, "qv");

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
