#include "SPMT.h"

int main(int argc, char** argv)
{
    SPMT* pmt = SPMT::Create(); 
    std::cout << pmt->desc() << std::endl ; 

    pmt->save("$FOLD"); 

    return 0 ; 
}
