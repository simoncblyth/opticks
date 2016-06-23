
#include "NGLM.hpp"

#include "GVector.hh"
#include "GMatrix.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"


void test_matrix()
{
    GMatrixF a ;
    a.Summary("a");

    GMatrixF b(
        2.f,0.f,0.f,0.f, 
        0.f,2.f,0.f,0.f, 
        0.f,0.f,2.f,0.f, 
        0.f,0.f,0.f,2.f);

    b.Summary("b");
     
    GMatrixF c(
        0.f,0.f,0.f,1.f, 
        0.f,0.f,1.f,0.f, 
        0.f,1.f,0.f,0.f, 
        1.f,0.f,0.f,0.f);

    c.Summary("c");
     


    GMatrixF p ;
    p *= a ;
    p *= b ;
    p *= c ;

    p.Summary("p");

    GMatrixF t(
        0.f,0.f,0.f,10.f, 
        0.f,0.f,0.f,20.f, 
        0.f,0.f,0.f,30.f, 
        0.f,0.f,0.f,1.f);

    gfloat3 v(0.f,0.f,0.f);
    v *= t ; 
    v.Summary("v");

}

void test_cf_glm()
{
     glm::vec3 sc(10.) ; 
     glm::vec3 tr(1.,2.,3.) ; 
    

     GMatrix<float> s(sc.x);
     s.Summary("s");     

     GMatrix<float> t(tr.x,tr.y,tr.z,sc.x);
     t.Summary("t");     

     GMatrix<float> tt((float*)t.getPointer());
     tt.Summary("tt");


     glm::mat4 scale = glm::scale(glm::mat4(1.0f), sc);
     glm::mat4 translate = glm::translate(glm::mat4(1.0f), tr);
     glm::mat4 mat = translate * scale ; 

     GMatrix<float> gg(glm::value_ptr(mat));
     gg.Summary("gg");
}

void test_summary()
{
    GMatrix<float>* m = new GMatrix<float>(100.f, 200.f, 100.f,  10.f );
    m->Summary();

    std::cout << " sizeof(GMatrixF) " << sizeof(GMatrixF)
              << " sizeof(float)*4*4 " << sizeof(float)*4*4 
              << std::endl 
              ;
}




int main(int argc, char** argv)
{
     PLOG_(argc, argv);
     GGEO_LOG_ ;  

     test_matrix();
     test_cf_glm();
     test_summary();


     return 0 ; 
}
