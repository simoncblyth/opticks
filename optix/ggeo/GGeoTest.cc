#include "GMaterial.hh"
#include "GProperty.hh"

#include <string>
#include <vector>


int main(int argc, char* argv[])
{
    GMaterial* mat = new GMaterial("demo");

    float domain[]={1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    float vals[]  ={10.f,20.f,30.f,40.f,50.f,60.f,70.f};

    mat->AddProperty("pname", vals, domain, sizeof(domain)/sizeof(domain[0]) );

    GProperty<float>* prop = mat->GetProperty("pname");
    prop->Summary("prop dump");


    return 0 ;
}


