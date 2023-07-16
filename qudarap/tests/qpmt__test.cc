
#include "SPMT.h"
#include "NPFold.h"

#include "qpmt.h"

int main()
{
    SPMT* spmt = SPMT::Load();
    if(spmt == nullptr) return 1 ; 

    NPFold* spmt_f = spmt->serialize() ; 
    std::cout << spmt_f->desc() ;  

    



    return 0 ; 
}
