#include "SPMT.h"

int main(int argc, char** argv)
{
    SPMT* pmt = SPMT::Load(); 
    if(pmt == nullptr) return 1 ; 

    std::cout << pmt->desc() << std::endl ; 
    pmt->save("$SFOLD"); 

    NPFold* f = pmt->get_ARTE(); 
    f->save("$SFOLD/get_ARTE/xscan") ; 

    return 0 ; 
}
