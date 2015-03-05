#include "GMaterial.hh"
#include "GProperty.hh"
#include "GVector.hh"
#include "GMatrix.hh"

#include <string>
#include <vector>


int main(int argc, char* argv[])
{

    /*
    GMaterial* mat = new GMaterial("demo", 0);

    float domain[]={1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    float vals[]  ={10.f,20.f,30.f,40.f,50.f,60.f,70.f};

    mat->AddProperty("pname", vals, domain, sizeof(domain)/sizeof(domain[0]) );

    GProperty<float>* prop = mat->GetProperty("pname");
    prop->Summary("prop dump");

    */

    GMatrixF a ;
    a.Dump("a");


    GMatrixF b(
        2.f,0.f,0.f,0.f, 
        0.f,2.f,0.f,0.f, 
        0.f,0.f,2.f,0.f, 
        0.f,0.f,0.f,2.f);

    b.Dump("b");
     
    GMatrixF c(
        0.f,0.f,0.f,1.f, 
        0.f,0.f,1.f,0.f, 
        0.f,1.f,0.f,0.f, 
        1.f,0.f,0.f,0.f);

    c.Dump("c");
     


    GMatrixF p ;
    p *= a ;
    p *= b ;
    p *= c ;

    p.Dump("p");

    GMatrixF t(
        0.f,0.f,0.f,10.f, 
        0.f,0.f,0.f,20.f, 
        0.f,0.f,0.f,30.f, 
        0.f,0.f,0.f,1.f);

    gfloat3 v(0.f,0.f,0.f);
    v *= t ; 
    v.Dump("v");



    return 0 ;
}


