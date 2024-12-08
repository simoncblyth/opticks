#include "SPMT.h"

/*
struct SPMT_test
{
};
*/


int main(int argc, char** argv)
{
    SPMT* pmt = SPMT::Load(); 
    if(pmt == nullptr) return 1 ; 

    std::cout << pmt->desc() << std::endl ; 
    NPFold* spmt_f = pmt->serialize(); 
    spmt_f->save("$SFOLD/spmt"); 

    NPFold* sscan = pmt->make_sscan(); 
    sscan->save("$SFOLD/sscan") ; 

    return 0 ; 
}
