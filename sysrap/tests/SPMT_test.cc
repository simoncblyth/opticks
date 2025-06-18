#include "SPMT.h"

int main(int argc, char** argv)
{
    SPMT* pmt = SPMT::CreateFromJPMT();
    if(pmt == nullptr) return 1 ;

    std::cout << pmt->desc() << std::endl ;
    NPFold* spmt_f = pmt->serialize();
    spmt_f->save("$SFOLD/spmt");

    //NPFold* c4scan = pmt->make_c4scan();
    //c4scan->save("$SFOLD/c4scan") ;

    NPFold* testfold = pmt->make_testfold();
    testfold->save("$SFOLD/testfold") ;


    return 0 ;
}
