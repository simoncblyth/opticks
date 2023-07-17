/**
QPMT_MockTest.cc
==================

**/


#include "SPMT.h"
#include "NPFold.h"
#include "QPMTTest.h"

int main()
{
    SPMT* spmt = SPMT::Load();
    if(spmt == nullptr) return 1 ; 

    NPFold* jpmt = spmt->serialize() ; 
    std::cout << jpmt->desc() ;  

    QPMTTest<float> t(jpmt); 
    NPFold* f = t.serialize(); 
    f->save("$FOLD"); 

    return 0 ; 
}
