#include "SPMT.h"

int main(int argc, char** argv)
{
    SPMT* pmt = SPMT::CreateFromJPMT();
    if(pmt == nullptr) return 1 ;

    std::cout << pmt->desc() << std::endl ;
    NPFold* spmt_f = pmt->serialize();
    spmt_f->save("$FOLD/spmt");

    NPFold* testfold = pmt->make_testfold();
    testfold->save("$FOLD/testfold") ;


    return 0 ;
}
