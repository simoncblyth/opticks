#include "GMaterial.hh"
#include "GProperty.hh"
#include "GVector.hh"
#include "GMatrix.hh"
#include "GBoundaryLib.hh"

#include <string>
#include <vector>


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



void test_material()
{
    GMaterial* mat = new GMaterial("demo", 0);

    float domain[]={1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    float vals[]  ={10.f,20.f,30.f,40.f,50.f,60.f,70.f};

    mat->addProperty("pname", vals, domain, sizeof(domain)/sizeof(domain[0]) );

    GProperty<float>* prop = mat->getProperty("pname");
    prop->Summary("prop dump");
}


void test_substancelib()
{
    GBoundaryLib* lib = new GBoundaryLib();
    const char* ri = lib->getLocalKey("refractive_index");
    printf("ri %s \n", ri );
}




int main(int argc, char* argv[])
{

    GMatrixF* m = new GMatrixF(100.f, 200.f, 100.f,  10.f );
    m->Summary();

    printf(" size %lu   %lu \n", sizeof(GMatrixF), sizeof(float)*4*4 );
    


    return 0 ;
}


