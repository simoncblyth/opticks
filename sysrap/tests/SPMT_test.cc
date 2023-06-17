#include "SPMT.h"

int main(int argc, char** argv)
{
    SPMT* pmt = SPMT::Load(); 

    std::cout << pmt->desc() << std::endl ; 

    pmt->save("$FOLD"); 

    return 0 ; 
}
