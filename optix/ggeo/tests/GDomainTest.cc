#include "GDomain.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    GDomain<float>* dom = new GDomain<float>(0.f,100.f,11.f) ; 


    LOG(info) << " length " << dom->getLength() ; 

    dom->Summary(); 




    return 0 ; 
}

