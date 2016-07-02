
#include "GProperty.hh"
#include "GMaterial.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"



void test_zero()
{
    GMaterial* mat = new GMaterial("test", 0);
    mat->Summary(); 
}

void test_addProperty()
{
    GMaterial* mat = new GMaterial("demo", 0);

    float domain[]={1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    float vals[]  ={10.f,20.f,30.f,40.f,50.f,60.f,70.f};

    mat->addProperty("pname", vals, domain, sizeof(domain)/sizeof(domain[0]) );

    GProperty<float>* prop = mat->getProperty("pname");
    prop->Summary("prop dump");
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    test_zero();
    test_addProperty();


    return 0 ;
}

