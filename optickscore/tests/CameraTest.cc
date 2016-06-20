#include <cstdio>

#include "NGLM.hpp"
#include "GLMPrint.hpp"

#include "Camera.hh"

int main()
{
    Camera* c = new Camera  ;

    float n(100.);
    float f(1000.);

    c->setNear(n);
    c->setFar(f);
    c->setParallel(false) ;

    c->Summary();

    glm::vec4 zpers ; 
    glm::vec4 zpara ;

    c->fillZProjection(zpers);

    c->setParallel(true) ;
    c->fillZProjection(zpara);


    float pers_22 = -(f+n)/(f-n) ;
    float pers_32 = -2.0f*f*n/(f-n) ;

    float para_22 = -2.0f/(f-n) ;
    float para_32 = -(f+n)/(f-n) ;


    print(zpers, "zpers");
    printf(" pers_22: -(f+n)/(f-n)  expect %10.4f  zpers.z  %10.4f \n", pers_22, zpers.z );   
    printf(" pers_32:   -2fn/(f-n)  expect %10.4f  zpers.w  %10.4f \n", pers_32, zpers.w );   

    print(zpara, "zpara");
    printf(" para_22:   -2/(f-n)    expect %10.4f  zpara.z  %10.4f diff %10.4f \n", para_22, zpara.z, para_22-zpara.z );   
    printf(" para_32:  -(f+n)/(f-n) expect %10.4f  zpers.w  %10.4f diff %10.4f \n", para_32, zpara.w, para_32-zpara.w );   


    // de-homogenizing does the perspective divide (dividing by -z)

    float ndcDepth_pers_n = -zpers.z - zpers.w/(-n) ;     // should be in range -1:1 for visibles
    float ndcDepth_pers_f = -zpers.z - zpers.w/(-f) ;     // should be in range -1:1 for visibles

    printf("(perspective) at eyeDist=-n or -f, ndcDepth %10.4f %10.4f \n", ndcDepth_pers_n, ndcDepth_pers_f  );

    // de-homogenizing in para case just divides by 1

    float ndcDepth_para_n = zpara.z*(-n) + zpara.w ;  
    float ndcDepth_para_f = zpara.z*(-f) + zpara.w ;    

    printf("(parallel)    at eyeDist=-n or -f, ndcDepth %10.4f %10.4f \n", ndcDepth_para_n, ndcDepth_para_f  );





}
